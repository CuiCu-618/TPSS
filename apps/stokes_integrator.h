/*
 * stokes_integrator.h
 *
 *  Created on: May 19, 2020
 *      Author: witte
 */

#ifndef APPS_STOKESINTEGRATOR_H_
#define APPS_STOKESINTEGRATOR_H_

#include <deal.II/base/subscriptor.h>

#include <deal.II/integrators/laplace.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>


#include "solvers_and_preconditioners/TPSS/fd_evaluation.h"
#include "solvers_and_preconditioners/TPSS/matrix_utilities.h"
#include "solvers_and_preconditioners/TPSS/tensor_product_matrix.h"


#include "biharmonic_integrator.h"
#include "common_integrator.h"
#include "equation_data.h"
#include "laplace_integrator.h"

namespace Stokes
{
using namespace dealii;

/**
 * Linear operators associated to the SIPG formulation (symmetric gradient) for
 * the stokes velocity with heterogeneous Dirichlet boundary conditions
 *
 * (MW) MeshWorker
 * (FD) FastDiagonalization
 */
namespace Velocity
{
namespace SIPG
{
// /**
//  * Standard (interior) penalty to obtain well-posedness of the Nitsche
//  * method. The penalty is weighted for face integrals at the physical
//  * boundary. The interior penalty is obtained by multiplying with 1/2.
//  */
// template<typename Number>
// Number
// compute_penalty_impl(const int degree, const Number h_left, const Number h_right)
// {
//   const auto one_over_h = (0.5 / h_left) + (0.5 / h_right);
//   const auto gamma      = degree == 0 ? 1 : degree * (degree + 1);
//   return 2.0 * gamma * one_over_h;
// }
using ::Nitsche::compute_penalty_impl;


namespace MW
{
using ::MW::compute_symgrad;

using ::MW::compute_average_symgrad;

using ::MW::compute_vvalue;

using ::MW::compute_vvalue_tangential;

using ::MW::compute_vjump;

using ::MW::compute_vjump_cross_normal;

using ::MW::compute_average_symgrad_tangential;

using ::MW::compute_vjump_tangential;

using ::MW::compute_vjump_cross_normal_tangential;

using ::MW::compute_vcurl;



template<int dim, bool with_shape_to_test>
struct ScratchDataSelector
{
  // empty. see specs.
};

template<int dim>
struct ScratchDataSelector<dim, false>
{
  using type = typename ::MW::StreamFunction::ScratchData<dim>;
};

template<int dim>
struct ScratchDataSelector<dim, true>
{
  using type = typename ::MW::TestFunction::ScratchData<dim>;
};

template<int dim, bool with_shape_to_test = false>
using ScratchData = typename ScratchDataSelector<dim, with_shape_to_test>::type;

using ::MW::DoF::CopyData;



using ::MW::StreamFunction::compute_symgrad;

using ::MW::StreamFunction::compute_vvalue;

using ::MW::StreamFunction::compute_vjump;

using ::MW::StreamFunction::compute_average_symgrad;

using ::MW::StreamFunction::compute_vjump_tangential;



using ::MW::TestFunction::compute_symgrad;

using ::MW::TestFunction::compute_vvalue;



using Biharmonic::Pressure::InterfaceId;

using Biharmonic::Pressure::InterfaceHandler;

template<int dim, typename CellIteratorType>
std::pair<std::vector<unsigned int>, std::vector<types::global_dof_index>>
get_active_interface_indices_impl(const InterfaceHandler<dim> & interface_handler,
                                  const CellIteratorType &      cell)
{
  std::pair<std::vector<unsigned int>, std::vector<types::global_dof_index>> indices;
  auto & [testfunc_indices, global_dof_indices_on_cell] = indices;

  for(auto face_no = 0U; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
  {
    const bool this_is_no_interface = cell->neighbor_index(face_no) == -1;
    if(this_is_no_interface)
      continue;

    const auto         ncell = cell->neighbor(face_no);
    InterfaceId        interface_id{cell->id(), ncell->id()};
    const unsigned int interface_index = interface_handler.get_interface_index(interface_id);

    const bool this_interface_isnt_contained = interface_index == numbers::invalid_unsigned_int;
    if(this_interface_isnt_contained)
      continue;

    testfunc_indices.push_back(face_no);
    global_dof_indices_on_cell.push_back(interface_index);
  }

  AssertDimension(testfunc_indices.size(), global_dof_indices_on_cell.size());
  return indices;
}



template<int dim, bool is_multigrid = false>
struct MatrixIntegrator
{
  using IteratorType = typename ::MW::IteratorSelector<dim, is_multigrid>::type;

  MatrixIntegrator(const Function<dim> *                              load_function_in,
                   const Function<dim> *                              analytical_solution_in,
                   const LinearAlgebra::distributed::Vector<double> * particular_solution,
                   const EquationData &                               equation_data_in,
                   const InterfaceHandler<dim> * interface_handler_in = nullptr)
    : load_function(load_function_in),
      analytical_solution(analytical_solution_in),
      discrete_solution(particular_solution),
      equation_data(equation_data_in),
      interface_handler(interface_handler_in)
  {
  }

  void
  cell_worker(const IteratorType & cell,
              ScratchData<dim> &   scratch_data,
              CopyData &           copy_data) const;

  void
  cell_worker_stream(const IteratorType & cell,
                     ScratchData<dim> &   scratch_data,
                     CopyData &           copy_data) const;

  void
  cell_residual_worker(const IteratorType &     cell_velocity,
                       const IteratorType &     cell_stream,
                       const IteratorType &     cell_pressure,
                       ScratchData<dim, true> & scratch_data,
                       CopyData &               copy_data) const;

  void
  cell_residual_worker_interface(const IteratorType &     cell_velocity,
                                 const IteratorType &     cell_stream,
                                 ScratchData<dim, true> & scratch_data,
                                 CopyData &               copy_data) const;

  template<typename TestEvaluatorType, typename AnsatzEvaluatorType>
  void
  cell_worker_impl(const TestEvaluatorType &   phi_test,
                   const AnsatzEvaluatorType & phi_ansatz,
                   CopyData::CellData &        copy_data) const;

  void
  face_worker(const IteratorType & cell,
              const unsigned int & f,
              const unsigned int & sf,
              const IteratorType & ncell,
              const unsigned int & nf,
              const unsigned int & nsf,
              ScratchData<dim> &   scratch_data,
              CopyData &           copy_data) const;

  void
  face_worker_stream(const IteratorType & cell,
                     const unsigned int & f,
                     const unsigned int & sf,
                     const IteratorType & ncell,
                     const unsigned int & nf,
                     const unsigned int & nsf,
                     ScratchData<dim> &   scratch_data,
                     CopyData &           copy_data) const;

  void
  face_worker_tangential(const IteratorType & cell,
                         const unsigned int & f,
                         const unsigned int & sf,
                         const IteratorType & ncell,
                         const unsigned int & nf,
                         const unsigned int & nsf,
                         ScratchData<dim> &   scratch_data,
                         CopyData &           copy_data) const;

  void
  face_residual_worker_tangential(const IteratorType &     cell,
                                  const IteratorType &     cell_stream,
                                  const IteratorType &     cell_pressure,
                                  const unsigned int &     f,
                                  const unsigned int &     sf,
                                  const IteratorType &     ncell,
                                  const IteratorType &     ncell_stream,
                                  const IteratorType &     ncell_pressure,
                                  const unsigned int &     nf,
                                  const unsigned int &     nsf,
                                  ScratchData<dim, true> & scratch_data,
                                  CopyData &               copy_data) const;

  void
  face_residual_worker_tangential_interface(const IteratorType &     cell,
                                            const IteratorType &     cell_stream,
                                            const unsigned int &     f,
                                            const unsigned int &     sf,
                                            const IteratorType &     ncell,
                                            const IteratorType &     ncell_stream,
                                            const unsigned int &     nf,
                                            const unsigned int &     nsf,
                                            ScratchData<dim, true> & scratch_data,
                                            CopyData &               copy_data) const;

  template<typename TestEvaluatorType, typename AnsatzEvaluatorType>
  void
  face_worker_impl(const TestEvaluatorType &   phi_test,
                   const AnsatzEvaluatorType & phi_ansatz,
                   const double                gamma_over_h,
                   CopyData::FaceData &        copy_data) const;

  template<typename TestEvaluatorType, typename AnsatzEvaluatorType>
  void
  face_worker_tangential_impl(const TestEvaluatorType &   phi_test,
                              const AnsatzEvaluatorType & phi_ansatz,
                              const double                gamma_over_h,
                              CopyData::FaceData &        copy_data) const;

  template<typename TestEvaluatorType, typename AnsatzEvaluatorType>
  void
  uniface_worker_impl(const TestEvaluatorType &   phi_test,
                      const AnsatzEvaluatorType & phi_ansatz,
                      const double                gamma_over_h,
                      CopyData::FaceData &        face_data) const;

  void
  uniface_worker(const IteratorType & cell,
                 const unsigned int & f,
                 const unsigned int & sf,
                 ScratchData<dim> &   scratch_data,
                 CopyData &           copy_data) const;

  void
  boundary_worker(const IteratorType & cell,
                  const unsigned int & face_no,
                  ScratchData<dim> &   scratch_data,
                  CopyData &           copy_data) const;

  void
  boundary_worker_stream(const IteratorType & cell,
                         const unsigned int & face_no,
                         ScratchData<dim> &   scratch_data,
                         CopyData &           copy_data) const;

  void
  boundary_worker_tangential(const IteratorType & cell,
                             const unsigned int & face_no,
                             ScratchData<dim> &   scratch_data,
                             CopyData &           copy_data) const;

  void
  boundary_residual_worker_tangential(const IteratorType &     cell,
                                      const IteratorType &     cell_stream,
                                      const IteratorType &     cell_pressure,
                                      const unsigned int &     face_no,
                                      ScratchData<dim, true> & scratch_data,
                                      CopyData &               copy_data) const;

  void
  boundary_residual_worker_tangential_interface(const IteratorType &     cell,
                                                const IteratorType &     cell_stream,
                                                const unsigned int &     face_no,
                                                ScratchData<dim, true> & scratch_data,
                                                CopyData &               copy_data) const;

  void
  uniface_worker_tangential(const IteratorType & cell,
                            const unsigned int & f,
                            const unsigned int & sf,
                            ScratchData<dim> &   scratch_data,
                            CopyData &           copy_data) const;

  template<bool do_rhs, typename TestEvaluatorType, typename AnsatzEvaluatorType>
  void
  boundary_worker_impl(const TestEvaluatorType &   phi_test,
                       const AnsatzEvaluatorType & phi_ansatz,
                       const double                gamma_over_h,
                       CopyData::FaceData &        copy_data) const;

  template<bool do_rhs, typename TestEvaluatorType, typename AnsatzEvaluatorType>
  void
  boundary_worker_tangential_impl(const TestEvaluatorType &   phi_test,
                                  const AnsatzEvaluatorType & phi_ansatz,
                                  const double                gamma_over_h,
                                  CopyData::FaceData &        copy_data) const;

  template<bool is_uniface>
  void
  boundary_or_uniface_worker_tangential_impl(const FEFaceValuesBase<dim> & phi_test,
                                             const FEFaceValuesBase<dim> & phi_ansatz,
                                             const double                  gamma_over_h,
                                             CopyData::FaceData &          face_data) const;

  // FREE FUNCTION
  /**
   * From the global dof indices on the left and right cell and the mapping of
   * joint dof indices to local cell dof indices on the left and right cell,
   * respectively, the associated global interface dof indices are computed.
   */
  std::vector<types::global_dof_index>
  get_interface_dof_indices(
    const std::vector<std::array<unsigned int, 2>> & joint_to_cell_dof_map,
    const std::vector<types::global_dof_index> &     dof_indices_on_left_cell,
    const std::vector<types::global_dof_index> &     dof_indices_on_right_cell) const;

  /**
   * We query the underlying InterfaceHandler to return the local-global pair of
   * interface indices associated to the current cell @p cell.
   */
  std::pair<std::vector<unsigned int>, std::vector<types::global_dof_index>>
  get_active_interface_indices(const IteratorType & cell) const;

  /**
   * From the global dof indices and its associated test function indices on the
   * left and right cell, respectively, the mapping of local joint dof indices
   * to local pairs of cell dof indices on the left and right cell is computed.
   */
  std::pair<std::vector<std::array<unsigned int, 2>>, std::vector<types::global_dof_index>>
  make_joint_interface_indices(
    const std::vector<unsigned int> &            testfunc_indices_left,
    const std::vector<types::global_dof_index> & dof_indices_on_lcell,
    const std::vector<unsigned int> &            testfunc_indices_right,
    const std::vector<types::global_dof_index> & dof_indices_on_rcell) const;

  const Function<dim> *                              load_function;
  const Function<dim> *                              analytical_solution;
  const LinearAlgebra::distributed::Vector<double> * discrete_solution;
  const EquationData                                 equation_data;
  const InterfaceHandler<dim> *                      interface_handler;
};



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::cell_worker(const IteratorType & cell,
                                                 ScratchData<dim> &   scratch_data,
                                                 CopyData &           copy_data) const
{
  AssertDimension(copy_data.cell_data.size(), 0U);

  FEValues<dim> & phi = scratch_data.fe_values_test;
  phi.reinit(cell);

  const unsigned int n_dofs_per_cell = phi.get_fe().dofs_per_cell;

  auto & cell_data = copy_data.cell_data.emplace_back(n_dofs_per_cell);

  cell->get_active_or_mg_dof_indices(cell_data.dof_indices);

  cell_worker_impl(phi, phi, cell_data);

  /// Subtract the particular solution @p discrete_solution from the right hand
  /// side, thus, as usual moving essential boundary conditions to the right
  /// hand side.
  ///
  /// If @p discrete_solution is not set (for example for a DG method) we skip
  /// here.
  if(!is_multigrid)
    if(discrete_solution && cell->at_boundary())
    {
      Vector<double> u0(cell_data.dof_indices.size());
      for(auto i = 0U; i < u0.size(); ++i)
        u0(i) = (*discrete_solution)(cell_data.dof_indices[i]);
      Vector<double> w0(cell_data.dof_indices.size());
      cell_data.matrix.vmult(w0, u0);
      cell_data.rhs -= w0;
    }
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::cell_worker_stream(const IteratorType & cell,
                                                        ScratchData<dim> &   scratch_data,
                                                        CopyData &           copy_data) const
{
  AssertDimension(copy_data.cell_data.size(), 0U);

  auto & phi = scratch_data.stream_values;
  phi.reinit(cell);

  auto & cell_data = copy_data.cell_data.emplace_back(phi.n_dofs_per_cell());

  cell->get_active_or_mg_dof_indices(cell_data.dof_indices);

  cell_worker_impl(phi, phi, cell_data);

  /// Subtract the particular solution @p discrete_solution from the right hand
  /// side, thus, as usual moving essential boundary conditions to the right
  /// hand side.
  ///
  /// If @p discrete_solution is not set (for example for a DG method) we skip
  /// here.
  if(!is_multigrid)
    if(discrete_solution && cell->at_boundary())
    {
      Vector<double> u0(cell_data.dof_indices.size());
      for(auto i = 0U; i < u0.size(); ++i)
        u0(i) = (*discrete_solution)(cell_data.dof_indices[i]);
      Vector<double> w0(cell_data.dof_indices.size());
      cell_data.matrix.vmult(w0, u0);
      cell_data.rhs -= w0;
    }
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::cell_residual_worker(const IteratorType &     cell,
                                                          const IteratorType &     cell_stream,
                                                          const IteratorType &     cell_pressure,
                                                          ScratchData<dim, true> & scratch_data,
                                                          CopyData &               copy_data) const
{
  AssertDimension(copy_data.cell_data.size(), 0U);

  auto & phi_test = scratch_data.test_values;
  phi_test.reinit(cell);

  auto & phi_ansatz = scratch_data.stream_values_ansatz;
  phi_ansatz.reinit(cell_stream);

  auto & cell_data =
    copy_data.cell_data.emplace_back(phi_test.n_dofs_per_cell(), phi_ansatz.n_dofs_per_cell());

  AssertDimension(phi_test.n_dofs_per_cell(), cell_data.dof_indices.size());

  std::vector<types::global_dof_index> dof_indices_pressure(cell_data.dof_indices.size() + 1);
  cell_pressure->get_active_or_mg_dof_indices(dof_indices_pressure);
  AssertDimension(dof_indices_pressure.size(), cell_data.dof_indices.size() + 1);
  /// skipping the first pressure dof
  std::copy(dof_indices_pressure.cbegin() + 1,
            dof_indices_pressure.cend(),
            cell_data.dof_indices.begin());

  cell_stream->get_active_or_mg_dof_indices(cell_data.dof_indices_column);

  cell_worker_impl(phi_test, phi_ansatz, cell_data);

  Assert(discrete_solution, ExcMessage("Stream function coefficients are not set."));

  Vector<double> dof_values(cell_data.dof_indices_column.size());
  std::transform(cell_data.dof_indices_column.cbegin(),
                 cell_data.dof_indices_column.cend(),
                 dof_values.begin(),
                 [&](const auto dof_index) { return (*discrete_solution)[dof_index]; });

  /// computing residual
  Vector<double> Ax(cell_data.rhs.size());
  cell_data.matrix.vmult(Ax, dof_values); // Ax
  cell_data.rhs -= Ax;                    // f - Ax

  cell_data.dof_indices_column.resize(2U);
  /// book-keeping index of first pressure dof
  cell_data.dof_indices_column.front() = dof_indices_pressure.front();
  /// book-keeping custom cell index
  cell_data.dof_indices_column.back() = interface_handler->get_cell_index(cell->id());
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::cell_residual_worker_interface(
  const IteratorType &     cell,
  const IteratorType &     cell_stream,
  ScratchData<dim, true> & scratch_data,
  CopyData &               copy_data) const
{
  AssertDimension(copy_data.cell_data.size(), 0U);

  /// The InterfaceHandler determines for which "inflow" interfaces this cell is
  /// the source cell (and in that sense we speak of "outflow" interfaces"): the
  /// local face numbers and the global interface indices are given by the first
  /// and second return value, respectively.
  auto [active_test_function_indices, global_interface_indices] =
    get_active_interface_indices(cell);

  auto & phi_test = scratch_data.test_values;
  /// Restricting TestFunction::Values to those test functions which are active
  /// on associated "outflow" interfaces for this cell.
  phi_test.reinit(cell, active_test_function_indices);

  auto & phi_ansatz = scratch_data.stream_values_ansatz;
  phi_ansatz.reinit(cell_stream);

  AssertDimension(phi_test.shape_to_test_functions.m(), GeometryInfo<dim>::faces_per_cell);

  auto & cell_data =
    copy_data.cell_data.emplace_back(phi_test.n_dofs_per_cell(), phi_ansatz.n_dofs_per_cell());

  std::swap(cell_data.dof_indices, global_interface_indices);

  cell_stream->get_active_or_mg_dof_indices(cell_data.dof_indices_column);

  cell_worker_impl(phi_test, phi_ansatz, cell_data);

  AssertDimension(cell_data.matrix.n(), cell_data.dof_indices_column.size());
  Assert(discrete_solution, ExcMessage("Stream function coefficients are not set."));
  Vector<double> dof_values(cell_data.dof_indices_column.size());
  std::transform(cell_data.dof_indices_column.cbegin(),
                 cell_data.dof_indices_column.cend(),
                 dof_values.begin(),
                 [&](const auto dof_index) { return (*discrete_solution)[dof_index]; });

  AssertDimension(cell_data.matrix.m(), cell_data.rhs.size());
  Vector<double> Ax(cell_data.rhs.size());
  cell_data.matrix.vmult(Ax, dof_values); // Ax
  cell_data.rhs -= Ax;                    // f - Ax
}



template<int dim, bool is_multigrid>
template<typename TestEvaluatorType, typename AnsatzEvaluatorType>
void
MatrixIntegrator<dim, is_multigrid>::cell_worker_impl(const TestEvaluatorType &   phi_test,
                                                      const AnsatzEvaluatorType & phi_ansatz,
                                                      CopyData::CellData &        cell_data) const
{
  AssertDimension(cell_data.matrix.m(), cell_data.rhs.size());
  AssertDimension(cell_data.matrix.m(), cell_data.dof_indices.size());
  AssertDimension(cell_data.matrix.n(),
                  cell_data.dof_indices_column.empty() ? cell_data.dof_indices.size() :
                                                         cell_data.dof_indices_column.size());

  std::vector<Tensor<1, dim>> load_values;
  if(!is_multigrid)
  {
    Assert(load_function, ExcMessage("load_function is not set."));
    AssertDimension(load_function->n_components, dim);
    const auto & q_points = phi_test.get_quadrature_points();
    std::transform(q_points.cbegin(),
                   q_points.cend(),
                   std::back_inserter(load_values),
                   [this](const auto & x_q) {
                     Tensor<1, dim> value;
                     for(auto c = 0U; c < dim; ++c)
                       value[c] = load_function->value(x_q, c);
                     return value;
                   });
  }

  for(unsigned int q = 0; q < phi_test.n_quadrature_points; ++q)
  {
    for(unsigned int i = 0; i < cell_data.matrix.m(); ++i)
    {
      const SymmetricTensor<2, dim> symgrad_phi_i = compute_symgrad(phi_test, i, q);
      for(unsigned int j = 0; j < cell_data.matrix.n(); ++j)
      {
        const SymmetricTensor<2, dim> symgrad_phi_j = compute_symgrad(phi_ansatz, j, q);

        cell_data.matrix(i, j) += 2. *
                                  scalar_product(symgrad_phi_i,   // symgrad phi_i(x)
                                                 symgrad_phi_j) * // symgrad phi_j(x)
                                  phi_test.JxW(q);                // dx
      }

      if(!is_multigrid)
      {
        const auto & phi_i = compute_vvalue(phi_test, i, q);
        cell_data.rhs(i) += phi_i * load_values[q] * phi_test.JxW(q);
      }
    }
  }
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::face_worker(const IteratorType & cell,
                                                 const unsigned int & f,
                                                 const unsigned int & sf,
                                                 const IteratorType & ncell,
                                                 const unsigned int & nf,
                                                 const unsigned int & nsf,
                                                 ScratchData<dim> &   scratch_data,
                                                 CopyData &           copy_data) const
{
  FEInterfaceValues<dim> & fe_interface_values = scratch_data.fe_interface_values_test;
  fe_interface_values.reinit(cell, f, sf, ncell, nf, nsf);

  const unsigned int n_interface_dofs = fe_interface_values.n_current_interface_dofs();

  CopyData::FaceData & face_data = copy_data.face_data.emplace_back(n_interface_dofs);

  face_data.dof_indices        = fe_interface_values.get_interface_dof_indices();
  face_data.dof_indices_column = fe_interface_values.get_interface_dof_indices();

  const auto   h         = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f]);
  const auto   nh        = ncell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[nf]);
  const auto   fe_degree = scratch_data.fe_values_test.get_fe().degree;
  const double gamma_over_h =
    equation_data.ip_factor * 0.5 * compute_penalty_impl(fe_degree, h, nh);

  face_worker_impl(fe_interface_values, fe_interface_values, gamma_over_h, face_data);

  AssertDimension(face_data.matrix.m(), n_interface_dofs);
  AssertDimension(face_data.matrix.n(), n_interface_dofs);
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::face_worker_stream(const IteratorType & cell,
                                                        const unsigned int & f,
                                                        const unsigned int & sf,
                                                        const IteratorType & ncell,
                                                        const unsigned int & nf,
                                                        const unsigned int & nsf,
                                                        ScratchData<dim> &   scratch_data,
                                                        CopyData &           copy_data) const
{
  auto & phi = scratch_data.stream_interface_values;
  phi.reinit(cell, f, sf, ncell, nf, nsf);

  const unsigned int   n_interface_dofs = phi.n_current_interface_dofs();
  CopyData::FaceData & face_data        = copy_data.face_data.emplace_back(n_interface_dofs);

  face_data.dof_indices = phi.get_interface_dof_indices();
  // face_data.dof_indices_column = face_data.dof_indices;

  // copy_data_face.cell_matrix.reinit(n_interface_dofs, n_interface_dofs);

  const auto   h         = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f]);
  const auto   nh        = ncell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[nf]);
  const auto   fe_degree = scratch_data.fe_values_test.get_fe().degree;
  const double gamma_over_h =
    equation_data.ip_factor * 0.5 * compute_penalty_impl(fe_degree, h, nh);

  face_worker_impl(phi, phi, gamma_over_h, face_data);

  AssertDimension(face_data.matrix.m(), n_interface_dofs);
  AssertDimension(face_data.matrix.n(), n_interface_dofs);
}



template<int dim, bool is_multigrid>
template<typename TestEvaluatorType, typename AnsatzEvaluatorType>
void
MatrixIntegrator<dim, is_multigrid>::face_worker_impl(const TestEvaluatorType &   phi_test,
                                                      const AnsatzEvaluatorType & phi_ansatz,
                                                      const double                gamma_over_h,
                                                      CopyData::FaceData &        face_data) const
{
  const auto n_interface_dofs_test   = phi_test.n_current_interface_dofs();
  const auto n_interface_dofs_ansatz = phi_ansatz.n_current_interface_dofs();

  AssertDimension(face_data.matrix.m(), n_interface_dofs_test);
  AssertDimension(face_data.matrix.n(), n_interface_dofs_ansatz);

  double integral_ijq = 0.;
  for(unsigned int q = 0; q < phi_test.n_quadrature_points; ++q)
  {
    const auto & n = phi_test.normal(q);
    for(unsigned int i = 0; i < n_interface_dofs_test; ++i)
    {
      const auto & av_symgrad_phi_i = compute_average_symgrad(phi_test, i, q);
      const auto & jump_phi_i       = compute_vjump(phi_test, i, q);
      // TODO !!!
      // Due to the symmetry of the average symgrad it is not important if
      //      [[ phi ]] (x) n
      // OR   n (x) [[ phi ]]
      // BUT for tangential worker it might be relevant...
      const auto & jump_phi_i_cross_n = outer_product(jump_phi_i, n);

      for(unsigned int j = 0; j < n_interface_dofs_ansatz; ++j)
      {
        const auto & av_symgrad_phi_j   = compute_average_symgrad(phi_ansatz, j, q);
        const auto & jump_phi_j         = compute_vjump(phi_ansatz, j, q);
        const auto & jump_phi_j_cross_n = outer_product(jump_phi_j, n);

        integral_ijq = -scalar_product(av_symgrad_phi_j, jump_phi_i_cross_n);
        integral_ijq += -scalar_product(jump_phi_j_cross_n, av_symgrad_phi_i);
        integral_ijq += gamma_over_h * jump_phi_j * jump_phi_i;
        integral_ijq *= 2. * phi_test.JxW(q);

        face_data.matrix(i, j) += integral_ijq;
      }
    }
  }
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::face_worker_tangential(const IteratorType & cell,
                                                            const unsigned int & f,
                                                            const unsigned int & sf,
                                                            const IteratorType & ncell,
                                                            const unsigned int & nf,
                                                            const unsigned int & nsf,
                                                            ScratchData<dim> &   scratch_data,
                                                            CopyData &           copy_data) const
{
  FEInterfaceValues<dim> & fe_interface_values = scratch_data.fe_interface_values_test;
  fe_interface_values.reinit(cell, f, sf, ncell, nf, nsf);

  const unsigned int n_interface_dofs = fe_interface_values.n_current_interface_dofs();

  CopyData::FaceData & face_data = copy_data.face_data.emplace_back(n_interface_dofs);

  face_data.dof_indices = fe_interface_values.get_interface_dof_indices();

  const auto   h         = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f]);
  const auto   nh        = ncell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[nf]);
  const auto   fe_degree = scratch_data.fe_values_test.get_fe().degree;
  const double gamma_over_h =
    equation_data.ip_factor * 0.5 * compute_penalty_impl(fe_degree, h, nh);

  face_worker_tangential_impl(fe_interface_values, fe_interface_values, gamma_over_h, face_data);
}



template<int dim, bool is_multigrid>
template<typename TestEvaluatorType, typename AnsatzEvaluatorType>
void
MatrixIntegrator<dim, is_multigrid>::face_worker_tangential_impl(
  const TestEvaluatorType &   phi_test,
  const AnsatzEvaluatorType & phi_ansatz,
  const double                gamma_over_h,
  CopyData::FaceData &        face_data) const
{
  const auto n_interface_dofs_test   = phi_test.n_current_interface_dofs();
  const auto n_interface_dofs_ansatz = phi_ansatz.n_current_interface_dofs();

  AssertDimension(face_data.matrix.m(), n_interface_dofs_test);
  AssertDimension(face_data.matrix.n(), n_interface_dofs_ansatz);

  const std::vector<Tensor<1, dim>> & normals = phi_test.get_normal_vectors();

  double integral_ijq = 0.;
  for(unsigned int q = 0; q < phi_test.n_quadrature_points; ++q)
  {
    const auto & n = normals[q];
    for(unsigned int i = 0; i < n_interface_dofs_test; ++i)
    {
      const auto & av_symgrad_phi_i = compute_average_symgrad(phi_test, i, q);
      const auto & jump_phit_i      = compute_vjump_tangential(phi_test, i, q);
      double       ncontrib_i       = n * av_symgrad_phi_i * n;
      // SymmetricTensor<2, dim> grad_phin_i;
      // for(unsigned int d = 0; d < dim; ++d)
      //   for(unsigned int dd = 0; dd < dim; ++dd)
      //     for(unsigned int k = 0; k < dim; ++k)
      //       grad_phin_i[d][dd] +=
      // 	      av_symgrad_phi_i[d][k] * n[k] * n[dd];
      //         // phi_test.average_gradient(i, q, k)[d] * n[k] * n[dd];

      for(unsigned int j = 0; j < n_interface_dofs_ansatz; ++j)
      {
        const auto & av_symgrad_phi_j = compute_average_symgrad(phi_ansatz, j, q);
        const auto & jump_phit_j      = compute_vjump_tangential(phi_ansatz, j, q);
        double       ncontrib_j       = n * av_symgrad_phi_j * n;
        // SymmetricTensor<2, dim> grad_phin_j;
        // for(unsigned int d = 0; d < dim; ++d)
        //   for(unsigned int dd = 0; dd < dim; ++dd)
        //     for(unsigned int k = 0; k < dim; ++k)
        //       grad_phin_j[d][dd] +=
        // 	av_symgrad_phi_j[d][k] * n[k] * n[dd];
        //         // phi_test.average_gradient(j, q, k)[d] * n[k] * n[dd];

        integral_ijq = -(n * av_symgrad_phi_j - ncontrib_j * n) * jump_phit_i;
        integral_ijq += -(n * av_symgrad_phi_i - ncontrib_i * n) * jump_phit_j;
        // integral_ijq = -n * (av_symgrad_phi_j - grad_phin_j) * jump_phit_i;
        // integral_ijq += -n * (av_symgrad_phi_i - grad_phin_i) * jump_phit_j;
        integral_ijq += gamma_over_h * jump_phit_j * jump_phit_i;
        integral_ijq *= 2. * phi_test.JxW(q);

        face_data.matrix(i, j) += integral_ijq;
      }
    }
  }
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::face_residual_worker_tangential(
  const IteratorType &     cell,
  const IteratorType &     cell_stream,
  const IteratorType &     cell_pressure,
  const unsigned int &     face_no,
  const unsigned int &     subface_no,
  const IteratorType &     ncell,
  const IteratorType &     ncell_stream,
  const IteratorType &     ncell_pressure,
  const unsigned int &     nface_no,
  const unsigned int &     nsubface_no,
  ScratchData<dim, true> & scratch_data,
  CopyData &               copy_data) const
{
  auto & phi_test = scratch_data.test_interface_values;

  /// TODO to be removed
  const unsigned int n_test_functions_left  = phi_test.shape_to_test_functions_left.m();
  const unsigned int n_test_functions_right = phi_test.shape_to_test_functions_right.m();
  /// Test functions are a linear combination of shape functions belonging
  /// to cell-interior dofs, thus there are no joint dofs.
  std::vector<std::array<unsigned int, 2>> joint_testfunc_indices;
  for(auto li = 0U; li < n_test_functions_left; ++li)
    joint_testfunc_indices.push_back({li, numbers::invalid_unsigned_int});
  for(auto ri = 0U; ri < n_test_functions_right; ++ri)
    joint_testfunc_indices.push_back({numbers::invalid_unsigned_int, ri});
  // phi_test.reinit(cell, face_no, subface_no, ncell, nface_no, nsubface_no,
  // joint_testfunc_indices);
  phi_test.reinit(cell, face_no, subface_no, ncell, nface_no, nsubface_no);

  /// DEBUG
  // {
  //   std::ostringstream oss;
  //   oss << "auto: ";
  //   for(const auto & [li, ri] : phi_test.get_interface_test_function_indices())
  //     oss << " (" << li << "," << ri << ")";
  //   oss << std::endl;
  //   oss << "user: ";
  //   for(const auto & [li, ri] : joint_testfunc_indices)
  //     oss << " (" << li << "," << ri << ")";
  //   oss << std::endl;
  //   std::cout << oss.str() << std::endl;
  //   Assert(phi_test.get_interface_test_function_indices() == joint_testfunc_indices,
  //          ExcMessage("failed..."));
  // }

  auto & phi_ansatz = scratch_data.stream_interface_values_ansatz;
  phi_ansatz.reinit(cell_stream, face_no, subface_no, ncell_stream, nface_no, nsubface_no);

  CopyData::FaceData & face_data =
    copy_data.face_data.emplace_back(phi_test.n_current_interface_dofs(),
                                     phi_ansatz.n_current_interface_dofs());

  /// Test functions are constructed such that they have a 1-to-1 relation with each pressure dof
  /// except the constant pressure mode (which is the first for Legendre type finite elements).
  const auto & make_dof_indices_skipping_first = [](const auto & cell) {
    std::vector<types::global_dof_index> all(cell->get_fe().dofs_per_cell);
    cell->get_active_or_mg_dof_indices(all);
    return std::vector<types::global_dof_index>(all.begin() + 1, all.end());
  };
  const auto dof_indices_on_lcell_pressure =
    std::move(make_dof_indices_skipping_first(cell_pressure));
  const auto dof_indices_on_rcell_pressure =
    std::move(make_dof_indices_skipping_first(ncell_pressure));
  face_data.dof_indices =
    std::move(get_interface_dof_indices(phi_test.get_interface_test_function_indices(),
                                        dof_indices_on_lcell_pressure,
                                        dof_indices_on_rcell_pressure));

  face_data.dof_indices_column = std::move(phi_ansatz.get_interface_dof_indices());

  AssertDimension(face_data.matrix.m(), face_data.dof_indices.size());
  AssertDimension(face_data.matrix.n(), face_data.dof_indices_column.size());

  const auto   h  = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[face_no]);
  const auto   nh = ncell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[nface_no]);
  const auto   fe_degree = phi_ansatz.get_fe().degree; // stream function degree !
  const double gamma_over_h =
    equation_data.ip_factor * 0.5 * compute_penalty_impl(fe_degree, h, nh);

  face_worker_tangential_impl(phi_test, phi_ansatz, gamma_over_h, face_data);

  Assert(discrete_solution, ExcMessage("Discrete stream function solution isnt set."));
  Vector<double> dof_values(face_data.dof_indices_column.size());
  std::transform(face_data.dof_indices_column.cbegin(),
                 face_data.dof_indices_column.cend(),
                 dof_values.begin(),
                 [&](const auto dof_index) { return (*discrete_solution)[dof_index]; });

  AssertDimension(face_data.matrix.n(), dof_values.size());
  AssertDimension(face_data.matrix.m(), face_data.rhs.size());
  Vector<double> Ax(face_data.rhs.size());
  face_data.matrix.vmult(Ax, dof_values); // Ax
  face_data.rhs -= Ax;                    // f - Ax
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::face_residual_worker_tangential_interface(
  const IteratorType &     cell,
  const IteratorType &     cell_stream,
  const unsigned int &     f,
  const unsigned int &     sf,
  const IteratorType &     ncell,
  const IteratorType &     ncell_stream,
  const unsigned int &     nf,
  const unsigned int &     nsf,
  ScratchData<dim, true> & scratch_data,
  CopyData &               copy_data) const
{
  const InterfaceId interface_id{cell->id(), ncell->id()};
  const auto        interface_index        = interface_handler->get_interface_index(interface_id);
  const bool this_interface_isnt_contained = interface_index == numbers::invalid_unsigned_int;

  if(this_interface_isnt_contained)
    return;

  const auto & [active_test_function_indices_left, global_interface_indices_lcell] =
    get_active_interface_indices(cell);
  const auto & [active_test_function_indices_right, global_interface_indices_rcell] =
    get_active_interface_indices(ncell);
  /// TODO do test functions associated with other faces than this interface
  /// contribute to the integral over this interface???
  auto [active_interface_test_function_indices, joint_dof_indices_test] =
    make_joint_interface_indices(active_test_function_indices_left,
                                 global_interface_indices_lcell,
                                 active_test_function_indices_right,
                                 global_interface_indices_rcell);

  auto & phi_test = scratch_data.test_interface_values;
  phi_test.reinit(cell, f, sf, ncell, nf, nsf, active_interface_test_function_indices);

  AssertDimension(phi_test.shape_to_test_functions_left.m(), GeometryInfo<dim>::faces_per_cell);
  AssertDimension(phi_test.shape_to_test_functions_right.m(), GeometryInfo<dim>::faces_per_cell);

  auto & phi_ansatz = scratch_data.stream_interface_values_ansatz;
  phi_ansatz.reinit(cell_stream, f, sf, ncell_stream, nf, nsf);

  CopyData::FaceData & face_data =
    copy_data.face_data.emplace_back(phi_test.n_current_interface_dofs(),
                                     phi_ansatz.n_current_interface_dofs());

  std::swap(face_data.dof_indices, joint_dof_indices_test);

  face_data.dof_indices_column = phi_ansatz.get_interface_dof_indices();

  const auto   h         = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f]);
  const auto   nh        = ncell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[nf]);
  const auto   fe_degree = phi_ansatz.get_fe().degree; // stream function ansatz !
  const double gamma_over_h =
    equation_data.ip_factor * 0.5 * compute_penalty_impl(fe_degree, h, nh);

  face_worker_tangential_impl(phi_test, phi_ansatz, gamma_over_h, face_data);

  AssertDimension(face_data.matrix.m(), face_data.rhs.size());
  AssertDimension(face_data.matrix.n(), face_data.dof_indices_column.size());

  Assert(discrete_solution, ExcMessage("Stream function coefficients are not set."));
  Vector<double> dof_values(face_data.dof_indices_column.size());
  std::transform(face_data.dof_indices_column.cbegin(),
                 face_data.dof_indices_column.cend(),
                 dof_values.begin(),
                 [&](const auto dof_index) { return (*discrete_solution)[dof_index]; });

  Vector<double> Ax(face_data.rhs.size());
  face_data.matrix.vmult(Ax, dof_values); // Ax
  face_data.rhs -= Ax;                    // f - Ax
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::boundary_worker(const IteratorType & cell,
                                                     const unsigned int & f,
                                                     ScratchData<dim> &   scratch_data,
                                                     CopyData &           copy_data) const
{
  FEInterfaceValues<dim> & fe_interface_values = scratch_data.fe_interface_values_test;
  fe_interface_values.reinit(cell, f);

  const unsigned int n_dofs = fe_interface_values.n_current_interface_dofs();

  CopyData::FaceData & face_data = copy_data.face_data.emplace_back(n_dofs);

  face_data.dof_indices        = fe_interface_values.get_interface_dof_indices();
  face_data.dof_indices_column = fe_interface_values.get_interface_dof_indices();

  const auto   h         = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f]);
  const auto   fe_degree = scratch_data.fe_values_test.get_fe().degree;
  const double gamma_over_h = equation_data.ip_factor * compute_penalty_impl(fe_degree, h, h);

  boundary_worker_impl<!is_multigrid>(fe_interface_values,
                                      fe_interface_values,
                                      gamma_over_h,
                                      face_data);

  AssertDimension(face_data.matrix.m(), face_data.dof_indices.size());
  AssertDimension(face_data.matrix.n(), face_data.dof_indices_column.size());
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::boundary_worker_stream(const IteratorType & cell,
                                                            const unsigned int & f,
                                                            ScratchData<dim> &   scratch_data,
                                                            CopyData &           copy_data) const
{
  auto & phi = scratch_data.stream_interface_values;
  phi.reinit(cell, f);

  const unsigned int n_dofs = phi.n_current_interface_dofs();

  CopyData::FaceData & face_data = copy_data.face_data.emplace_back(n_dofs);

  face_data.dof_indices        = phi.get_interface_dof_indices();
  face_data.dof_indices_column = phi.get_interface_dof_indices();

  const auto   h         = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f]);
  const auto   fe_degree = scratch_data.fe_values_test.get_fe().degree;
  const double gamma_over_h = equation_data.ip_factor * compute_penalty_impl(fe_degree, h, h);

  boundary_worker_impl<!is_multigrid>(phi, phi, gamma_over_h, face_data);

  AssertDimension(face_data.matrix.m(), face_data.dof_indices.size());
  AssertDimension(face_data.matrix.n(), face_data.dof_indices_column.size());
}



template<int dim, bool is_multigrid>
template<bool do_rhs, typename TestEvaluatorType, typename AnsatzEvaluatorType>
void
MatrixIntegrator<dim, is_multigrid>::boundary_worker_impl(const TestEvaluatorType &   phi_test,
                                                          const AnsatzEvaluatorType & phi_ansatz,
                                                          const double                gamma_over_h,
                                                          CopyData::FaceData & face_data) const
{
  const auto n_interface_dofs_test   = phi_test.n_current_interface_dofs();
  const auto n_interface_dofs_ansatz = phi_ansatz.n_current_interface_dofs();

  AssertDimension(face_data.matrix.m(), n_interface_dofs_test);
  AssertDimension(face_data.matrix.n(), n_interface_dofs_ansatz);

  std::vector<Tensor<1, dim>>         solution_values;
  std::vector<Tensor<2, dim>>         solution_cross_normals;
  const std::vector<Tensor<1, dim>> & normals = phi_test.get_normal_vectors();
  if(do_rhs)
  {
    Assert(analytical_solution, ExcMessage("analytical_solution is not set."));
    AssertDimension(analytical_solution->n_components, dim);
    AssertDimension(face_data.rhs.size(), n_interface_dofs_test);

    const auto & q_points = phi_test.get_quadrature_points();
    std::transform(q_points.cbegin(),
                   q_points.cend(),
                   std::back_inserter(solution_values),
                   [this](const auto & x_q) {
                     Tensor<1, dim> value;
                     for(auto c = 0U; c < dim; ++c)
                       value[c] = analytical_solution->value(x_q, c);
                     return value;
                   });
    AssertDimension(normals.size(), solution_values.size());
    std::transform(solution_values.cbegin(),
                   solution_values.cend(),
                   normals.cbegin(),
                   std::back_inserter(solution_cross_normals),
                   [](const auto & u_q, const auto & normal) {
                     return outer_product(u_q, normal);
                   });
  }

  double integral_ijq = 0.;
  double nitsche_iq   = 0.;
  for(unsigned int q = 0; q < phi_test.n_quadrature_points; ++q)
  {
    const auto n = normals[q];
    for(unsigned int i = 0; i < n_interface_dofs_test; ++i)
    {
      const auto & av_symgrad_phi_i   = compute_average_symgrad(phi_test, i, q);
      const auto & jump_phi_i         = compute_vjump(phi_test, i, q);
      const auto & jump_phi_i_cross_n = outer_product(jump_phi_i, n);

      for(unsigned int j = 0; j < n_interface_dofs_ansatz; ++j)
      {
        const auto & av_symgrad_phi_j   = compute_average_symgrad(phi_ansatz, j, q);
        const auto & jump_phi_j         = compute_vjump(phi_ansatz, j, q);
        const auto & jump_phi_j_cross_n = outer_product(jump_phi_j, n);

        integral_ijq = -scalar_product(av_symgrad_phi_j, jump_phi_i_cross_n);
        integral_ijq += -scalar_product(jump_phi_j_cross_n, av_symgrad_phi_i);
        integral_ijq += gamma_over_h * jump_phi_j * jump_phi_i;
        integral_ijq *= 2. * phi_test.JxW(q);

        face_data.matrix(i, j) += integral_ijq;
      }

      /// Nitsche method (weak Dirichlet conditions)
      if(do_rhs)
      {
        const auto & u         = solution_values[q];
        const auto & u_cross_n = outer_product(u, n);

        nitsche_iq = -scalar_product(u_cross_n, av_symgrad_phi_i);
        nitsche_iq += gamma_over_h * u * jump_phi_i;
        nitsche_iq *= 2. * phi_test.JxW(q);

        face_data.rhs(i) += nitsche_iq;
      }
    }
  }
}



/// TODO use one implementation combining boundary_worker_impl and
/// uniface_worker_impl
template<int dim, bool is_multigrid>
template<typename TestEvaluatorType, typename AnsatzEvaluatorType>
void
MatrixIntegrator<dim, is_multigrid>::uniface_worker_impl(const TestEvaluatorType &   phi_test,
                                                         const AnsatzEvaluatorType & phi_ansatz,
                                                         const double                gamma_over_h,
                                                         CopyData::FaceData & face_data) const
{
  const auto n_interface_dofs_test   = phi_test.dofs_per_cell;
  const auto n_interface_dofs_ansatz = phi_ansatz.dofs_per_cell;

  AssertDimension(face_data.matrix.m(), n_interface_dofs_test);
  AssertDimension(face_data.matrix.n(), n_interface_dofs_ansatz);

  const std::vector<Tensor<1, dim>> & normals = phi_test.get_normal_vectors();

  double integral_ijq = 0.;
  for(unsigned int q = 0; q < phi_test.n_quadrature_points; ++q)
  {
    const auto n = normals[q];
    for(unsigned int i = 0; i < n_interface_dofs_test; ++i)
    {
      const auto & av_symgrad_phi_i   = 0.5 * compute_symgrad(phi_test, i, q);
      const auto & jump_phi_i         = compute_vvalue(phi_test, i, q);
      const auto & jump_phi_i_cross_n = outer_product(jump_phi_i, n);

      for(unsigned int j = 0; j < n_interface_dofs_ansatz; ++j)
      {
        const auto & av_symgrad_phi_j   = 0.5 * compute_symgrad(phi_ansatz, j, q);
        const auto & jump_phi_j         = compute_vvalue(phi_ansatz, j, q);
        const auto & jump_phi_j_cross_n = outer_product(jump_phi_j, n);

        integral_ijq = -scalar_product(av_symgrad_phi_j, jump_phi_i_cross_n);
        integral_ijq += -scalar_product(jump_phi_j_cross_n, av_symgrad_phi_i);
        integral_ijq += gamma_over_h * jump_phi_j * jump_phi_i;
        integral_ijq *= 2. * phi_test.JxW(q);

        face_data.matrix(i, j) += integral_ijq;
      }
    }
  }
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::uniface_worker(const IteratorType & cell,
                                                    const unsigned int & f,
                                                    const unsigned int & sf,
                                                    ScratchData<dim> &   scratch_data,
                                                    CopyData &           copy_data) const
{
  FEInterfaceValues<dim> & fe_interface_values = scratch_data.fe_interface_values_test;
  fe_interface_values.reinit(cell, f, sf, cell, f, sf);

  const FEFaceValuesBase<dim> & phi = fe_interface_values.get_fe_face_values(0);

  const unsigned int n_interface_dofs = phi.dofs_per_cell;

  CopyData::FaceData & face_data = copy_data.face_data.emplace_back(n_interface_dofs);

  cell->get_active_or_mg_dof_indices(face_data.dof_indices);
  face_data.dof_indices_column = face_data.dof_indices;

  const auto h = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f]);
  /// TODO In general we require the neighboring cell but it suffices to use h
  /// here (on distributed triangulation this leads to a global communication of
  /// h and nh).
  const auto   nh        = h;
  const auto   fe_degree = scratch_data.fe_values_test.get_fe().degree;
  const double gamma_over_h =
    equation_data.ip_factor * 0.5 * compute_penalty_impl(fe_degree, h, nh);

  uniface_worker_impl(phi, phi, gamma_over_h, face_data);

  AssertDimension(face_data.matrix.m(), n_interface_dofs);
  AssertDimension(face_data.matrix.n(), n_interface_dofs);
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::boundary_worker_tangential(const IteratorType & cell,
                                                                const unsigned int & f,
                                                                ScratchData<dim> &   scratch_data,
                                                                CopyData & copy_data) const
{
  FEInterfaceValues<dim> & fe_interface_values = scratch_data.fe_interface_values_test;
  fe_interface_values.reinit(cell, f);

  const unsigned int n_dofs = fe_interface_values.n_current_interface_dofs();

  CopyData::FaceData & face_data = copy_data.face_data.emplace_back(n_dofs);

  face_data.dof_indices = fe_interface_values.get_interface_dof_indices();

  const auto   h         = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f]);
  const auto   fe_degree = scratch_data.fe_values_test.get_fe().degree;
  const double gamma_over_h = equation_data.ip_factor * compute_penalty_impl(fe_degree, h, h);

  boundary_worker_tangential_impl<!is_multigrid>(fe_interface_values,
                                                 fe_interface_values,
                                                 gamma_over_h,
                                                 face_data);
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::boundary_residual_worker_tangential(
  const IteratorType &     cell,
  const IteratorType &     cell_stream,
  const IteratorType &     cell_pressure,
  const unsigned int &     face_no,
  ScratchData<dim, true> & scratch_data,
  CopyData &               copy_data) const
{
  auto & phi_test = scratch_data.test_interface_values;

  const unsigned int n_test_functions_left = phi_test.shape_to_test_functions_left.m();

  // std::vector<std::array<unsigned int, 2>> joint_testfunc_indices;
  // for(auto li = 0U; li < n_test_functions_left; ++li)
  //   joint_testfunc_indices.push_back({li, numbers::invalid_unsigned_int});
  // phi_test.reinit(cell, face_no, joint_testfunc_indices);
  phi_test.reinit(cell, face_no);

  const auto                           n_dofs_per_cell_p = cell_pressure->get_fe().dofs_per_cell;
  std::vector<types::global_dof_index> dof_indices_on_lcell_pressure(n_dofs_per_cell_p);
  cell_pressure->get_active_or_mg_dof_indices(dof_indices_on_lcell_pressure);
  dof_indices_on_lcell_pressure.erase(dof_indices_on_lcell_pressure.begin());
  AssertDimension(dof_indices_on_lcell_pressure.size(), n_test_functions_left);

  auto & phi_ansatz = scratch_data.stream_interface_values_ansatz;
  phi_ansatz.reinit(cell_stream, face_no);

  CopyData::FaceData & face_data =
    copy_data.face_data.emplace_back(phi_test.n_current_interface_dofs(),
                                     phi_ansatz.n_current_interface_dofs());

  /// Test functions are constructed such that they have a 1-to-1 relation with each pressure dof
  /// except the constant pressure mode (which is the first for Legendre type finite elements).
  const auto & make_dof_indices_skipping_first = [](const auto & cell) {
    std::vector<types::global_dof_index> all(cell->get_fe().dofs_per_cell);
    cell->get_active_or_mg_dof_indices(all);
    return std::vector<types::global_dof_index>(all.begin() + 1, all.end());
  };
  face_data.dof_indices = std::move(make_dof_indices_skipping_first(cell_pressure));

  face_data.dof_indices_column = std::move(phi_ansatz.get_interface_dof_indices());

  const auto   h = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[face_no]);
  const auto   fe_degree    = phi_ansatz.get_fe().degree; // stream function degree !
  const double gamma_over_h = equation_data.ip_factor * compute_penalty_impl(fe_degree, h, h);

  boundary_worker_tangential_impl</*do_rhs*/ true>(phi_test, phi_ansatz, gamma_over_h, face_data);

  Assert(discrete_solution, ExcMessage("Discrete stream function solution isnt set."));
  Vector<double> dof_values(face_data.dof_indices_column.size());
  std::transform(face_data.dof_indices_column.cbegin(),
                 face_data.dof_indices_column.cend(),
                 dof_values.begin(),
                 [&](const auto dof_index) { return (*discrete_solution)[dof_index]; });

  AssertDimension(face_data.matrix.n(), dof_values.size());
  AssertDimension(face_data.matrix.m(), face_data.rhs.size());
  Vector<double> Ax(face_data.rhs.size());
  face_data.matrix.vmult(Ax, dof_values); // Ax
  face_data.rhs -= Ax;                    // f - Ax
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::boundary_residual_worker_tangential_interface(
  const IteratorType &     cell,
  const IteratorType &     cell_stream,
  const unsigned int &     face_no,
  ScratchData<dim, true> & scratch_data,
  CopyData &               copy_data) const
{
  auto [active_test_function_indices, global_interface_indices] =
    get_active_interface_indices(cell);
  std::vector<std::array<unsigned int, 2>> active_interface_test_function_indices;
  for(const auto li : active_test_function_indices)
    active_interface_test_function_indices.push_back({li, numbers::invalid_unsigned_int});

  auto & phi_test = scratch_data.test_interface_values;
  /// Restricting TestFunction::InterfaceValues to those test functions which are active
  /// on associated "outflow" interfaces for this cell.
  phi_test.reinit(cell, face_no, active_interface_test_function_indices);

  AssertDimension(phi_test.shape_to_test_functions_left.m(), GeometryInfo<dim>::faces_per_cell);
  AssertDimension(phi_test.shape_to_test_functions_right.m(), GeometryInfo<dim>::faces_per_cell);

  auto & phi_ansatz = scratch_data.stream_interface_values_ansatz;
  phi_ansatz.reinit(cell_stream, face_no);

  CopyData::FaceData & face_data =
    copy_data.face_data.emplace_back(phi_test.n_current_interface_dofs(),
                                     phi_ansatz.n_current_interface_dofs());

  face_data.dof_indices = std::move(global_interface_indices);

  AssertDimension(face_data.dof_indices.size(), active_interface_test_function_indices.size());

  face_data.dof_indices_column = phi_ansatz.get_interface_dof_indices();

  const auto   h = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[face_no]);
  const auto   fe_degree    = phi_ansatz.get_fe().degree; // stream function degree !
  const double gamma_over_h = equation_data.ip_factor * compute_penalty_impl(fe_degree, h, h);

  boundary_worker_tangential_impl</*do_rhs*/ true>(phi_test, phi_ansatz, gamma_over_h, face_data);

  AssertDimension(face_data.matrix.m(), face_data.rhs.size());
  AssertDimension(face_data.matrix.n(), face_data.dof_indices_column.size());

  Assert(discrete_solution, ExcMessage("Stream function coefficients are not set."));
  Vector<double> dof_values(face_data.dof_indices_column.size());
  std::transform(face_data.dof_indices_column.cbegin(),
                 face_data.dof_indices_column.cend(),
                 dof_values.begin(),
                 [&](const auto dof_index) { return (*discrete_solution)[dof_index]; });

  Vector<double> Ax(face_data.rhs.size());
  face_data.matrix.vmult(Ax, dof_values); // Ax
  face_data.rhs -= Ax;                    // f - Ax
}



template<int dim, bool is_multigrid>
template<bool do_rhs, typename TestEvaluatorType, typename AnsatzEvaluatorType>
void
MatrixIntegrator<dim, is_multigrid>::boundary_worker_tangential_impl(
  const TestEvaluatorType &   phi_test,
  const AnsatzEvaluatorType & phi_ansatz,
  const double                gamma_over_h,
  CopyData::FaceData &        face_data) const
{
  const auto n_interface_dofs_test   = phi_test.n_current_interface_dofs();
  const auto n_interface_dofs_ansatz = phi_ansatz.n_current_interface_dofs();

  AssertDimension(n_interface_dofs_test, face_data.matrix.m());
  AssertDimension(n_interface_dofs_ansatz, face_data.matrix.n());
  AssertDimension(n_interface_dofs_test, face_data.dof_indices.size());
  if(!face_data.dof_indices_column.empty())
    AssertDimension(n_interface_dofs_ansatz, face_data.dof_indices_column.size());

  const std::vector<Tensor<1, dim>> & normals = phi_test.get_normal_vectors();

  std::vector<Tensor<1, dim>> solution_values;
  std::vector<Tensor<1, dim>> tangential_solution_values;
  if(do_rhs)
  {
    Assert(analytical_solution, ExcMessage("analytical_solution is not set."));
    AssertDimension(analytical_solution->n_components, dim);
    const auto &                        q_points = phi_test.get_quadrature_points();
    const std::vector<Tensor<1, dim>> & normals  = phi_test.get_normal_vectors();
    std::transform(q_points.cbegin(),
                   q_points.cend(),
                   std::back_inserter(solution_values),
                   [this](const auto & x_q) {
                     Tensor<1, dim> value;
                     for(auto c = 0U; c < dim; ++c)
                       value[c] = analytical_solution->value(x_q, c);
                     return value;
                   });
    std::transform(solution_values.cbegin(),
                   solution_values.cend(),
                   normals.cbegin(),
                   std::back_inserter(tangential_solution_values),
                   [](const auto & u_q, const auto & normal) {
                     return u_q - ((u_q * normal) * normal);
                   });
    AssertDimension(solution_values.size(), phi_test.n_quadrature_points);
    AssertDimension(tangential_solution_values.size(), phi_test.n_quadrature_points);
  }

  double integral_ijq = 0.;
  double nitsche_iq   = 0.;
  for(unsigned int q = 0; q < phi_test.n_quadrature_points; ++q)
  {
    const auto & n = normals[q];
    for(unsigned int i = 0; i < n_interface_dofs_test; ++i)
    {
      const auto & jump_phit_i      = compute_vjump_tangential(phi_test, i, q);
      const auto & av_symgrad_phi_i = compute_average_symgrad(phi_test, i, q);
      double       ncontrib_i       = n * av_symgrad_phi_i * n;
      // SymmetricTensor<2, dim> grad_phin_i;
      // for(unsigned int d = 0; d < dim; ++d)
      //   for(unsigned int dd = 0; dd < dim; ++dd)
      //     for(unsigned int k = 0; k < dim; ++k)
      //       grad_phin_i[d][dd] +=
      //         phi_test.average_gradient(i, q, k)[d] * n[k] * n[dd];

      for(unsigned int j = 0; j < n_interface_dofs_ansatz; ++j)
      {
        const auto & jump_phit_j      = compute_vjump_tangential(phi_ansatz, j, q);
        const auto & av_symgrad_phi_j = compute_average_symgrad(phi_ansatz, j, q);
        double       ncontrib_j       = n * av_symgrad_phi_j * n;
        // SymmetricTensor<2, dim> grad_phin_j;
        // for(unsigned int d = 0; d < dim; ++d)
        //   for(unsigned int dd = 0; dd < dim; ++dd)
        //     for(unsigned int k = 0; k < dim; ++k)
        //       grad_phin_j[d][dd] +=
        //         phi_test.average_gradient(j, q, k)[d] * n[k] * n[dd];

        integral_ijq = -(n * av_symgrad_phi_j - ncontrib_j * n) * jump_phit_i;
        integral_ijq += -(n * av_symgrad_phi_i - ncontrib_i * n) * jump_phit_j;
        // integral_ijq = -n * (av_symgrad_phi_j - grad_phin_j) * jump_phit_i;
        // integral_ijq += -n * (av_symgrad_phi_i - grad_phin_i) * jump_phit_j;
        integral_ijq += gamma_over_h * jump_phit_j * jump_phit_i;
        integral_ijq *= 2. * phi_test.JxW(q);

        face_data.matrix(i, j) += integral_ijq;
      }

      /// Nitsche method (weak Dirichlet conditions)
      if(do_rhs)
      {
        /// ut is the tangential vector field of the vector field u (which
        /// should not be confused with the tangential component of the vector
        /// field u!).
        const auto & ut = tangential_solution_values[q];

        nitsche_iq = -(n * av_symgrad_phi_i - n * ncontrib_i) * ut;
        // nitsche_iq = -n * (av_symgrad_phi_i - grad_phin_i) * ut;
        nitsche_iq += gamma_over_h * ut * jump_phit_i;
        nitsche_iq *= 2. * phi_test.JxW(q);

        face_data.rhs(i) += nitsche_iq;
      }
    }
  }
}



template<int dim, bool is_multigrid>
template<bool is_uniface>
void
MatrixIntegrator<dim, is_multigrid>::boundary_or_uniface_worker_tangential_impl(
  const FEFaceValuesBase<dim> & phi_test,
  const FEFaceValuesBase<dim> & phi_ansatz,
  const double                  gamma_over_h,
  CopyData::FaceData &          face_data) const
{
  constexpr bool do_rhs = !is_multigrid && !is_uniface;

  const auto n_interface_dofs_test   = face_data.matrix.m();
  const auto n_interface_dofs_ansatz = face_data.matrix.n();

  AssertDimension(n_interface_dofs_test, face_data.dof_indices.size());
  if(!face_data.dof_indices_column.empty())
    AssertDimension(n_interface_dofs_ansatz, face_data.dof_indices_column.size());

  const std::vector<Tensor<1, dim>> & normals = phi_test.get_normal_vectors();

  std::vector<Tensor<1, dim>> solution_values;
  std::vector<Tensor<1, dim>> tangential_solution_values;
  if(do_rhs)
  {
    Assert(analytical_solution, ExcMessage("analytical_solution is not set."));
    AssertDimension(analytical_solution->n_components, dim);
    const auto &                        q_points = phi_test.get_quadrature_points();
    const std::vector<Tensor<1, dim>> & normals  = phi_test.get_normal_vectors();
    std::transform(q_points.cbegin(),
                   q_points.cend(),
                   std::back_inserter(solution_values),
                   [this](const auto & x_q) {
                     Tensor<1, dim> value;
                     for(auto c = 0U; c < dim; ++c)
                       value[c] = analytical_solution->value(x_q, c);
                     return value;
                   });
    std::transform(solution_values.cbegin(),
                   solution_values.cend(),
                   normals.cbegin(),
                   std::back_inserter(tangential_solution_values),
                   [](const auto & u_q, const auto & normal) {
                     return u_q - ((u_q * normal) * normal);
                   });
    AssertDimension(solution_values.size(), phi_test.n_quadrature_points);
    AssertDimension(tangential_solution_values.size(), phi_test.n_quadrature_points);
  }

  double integral_ijq = 0.;
  double nitsche_iq   = 0.;
  for(unsigned int q = 0; q < phi_test.n_quadrature_points; ++q)
  {
    const auto & n = normals[q];
    for(unsigned int i = 0; i < n_interface_dofs_test; ++i)
    {
      const auto & jump_phit_i      = compute_vvalue_tangential(phi_test, i, q);
      const auto & av_symgrad_phi_i = (is_uniface ? 0.5 : 1.0) * compute_symgrad(phi_test, i, q);
      double       ncontrib_i       = n * av_symgrad_phi_i * n;

      for(unsigned int j = 0; j < n_interface_dofs_ansatz; ++j)
      {
        const auto & jump_phit_j = compute_vvalue_tangential(phi_ansatz, j, q);
        const auto & av_symgrad_phi_j =
          (is_uniface ? 0.5 : 1.0) * compute_symgrad(phi_ansatz, j, q);
        double ncontrib_j = n * av_symgrad_phi_j * n;

        integral_ijq = -(n * av_symgrad_phi_j - ncontrib_j * n) * jump_phit_i;
        integral_ijq += -(n * av_symgrad_phi_i - ncontrib_i * n) * jump_phit_j;
        integral_ijq += gamma_over_h * jump_phit_j * jump_phit_i;
        integral_ijq *= 2. * phi_test.JxW(q);

        face_data.matrix(i, j) += integral_ijq;
      }

      /// Nitsche method (weak Dirichlet conditions)
      if(do_rhs)
      {
        /// ut is the tangential vector field of the vector field u (which
        /// should not be confused with the tangential component of the vector
        /// field u!).
        const auto & ut = tangential_solution_values[q];

        nitsche_iq = -(n * av_symgrad_phi_i - n * ncontrib_i) * ut;
        nitsche_iq += gamma_over_h * ut * jump_phit_i;
        nitsche_iq *= 2. * phi_test.JxW(q);

        face_data.rhs(i) += nitsche_iq;
      }
    }
  }
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::uniface_worker_tangential(const IteratorType & cell,
                                                               const unsigned int & f,
                                                               const unsigned int & sf,
                                                               ScratchData<dim> &   scratch_data,
                                                               CopyData &           copy_data) const
{
  scratch_data.fe_interface_values_test.reinit(cell, f, sf, cell, f, sf);
  const auto & phi = scratch_data.fe_interface_values_test.get_fe_face_values(0);

  const unsigned int n_dofs = phi.dofs_per_cell; // fe_interface_values.n_current_interface_dofs();

  CopyData::FaceData & face_data = copy_data.face_data.emplace_back(n_dofs);

  cell->get_active_or_mg_dof_indices(face_data.dof_indices);

  const auto h = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f]);
  /// TODO actually we need the characteristic width nh of the neigboring cell...
  const auto   nh        = h;
  const auto   fe_degree = scratch_data.fe_values_test.get_fe().degree;
  const double gamma_over_h =
    equation_data.ip_factor * 0.5 * compute_penalty_impl(fe_degree, h, nh);

  boundary_or_uniface_worker_tangential_impl<true>(phi, phi, gamma_over_h, face_data);
}



// FREE FUNCTION
template<int dim, bool is_multigrid>
std::vector<types::global_dof_index>
MatrixIntegrator<dim, is_multigrid>::get_interface_dof_indices(
  const std::vector<std::array<unsigned int, 2>> & joint_to_cell_dof_map,
  const std::vector<types::global_dof_index> &     dof_indices_on_left_cell,
  const std::vector<types::global_dof_index> &     dof_indices_on_right_cell) const
{
  AssertIndexRange(joint_to_cell_dof_map.size(),
                   dof_indices_on_left_cell.size() + dof_indices_on_right_cell.size() + 1);
  std::vector<types::global_dof_index> joint_dof_indices;
  for(auto j = 0U; j < joint_to_cell_dof_map.size(); ++j)
  {
    const auto [lj, rj] = joint_to_cell_dof_map[j];
    if(lj != numbers::invalid_unsigned_int)
      joint_dof_indices.emplace_back(dof_indices_on_left_cell[lj]);
    else if(rj != numbers::invalid_unsigned_int)
      joint_dof_indices.emplace_back(dof_indices_on_right_cell[rj]);
    else
      AssertThrow(false, ExcMessage("Check joint_to_cell_dof_map!"));
  }
  AssertDimension(joint_dof_indices.size(), joint_to_cell_dof_map.size());
  return joint_dof_indices;
}


template<int dim, bool is_multigrid>
std::pair<std::vector<unsigned int>, std::vector<types::global_dof_index>>
MatrixIntegrator<dim, is_multigrid>::get_active_interface_indices(const IteratorType & cell) const
{
  return get_active_interface_indices_impl(*interface_handler, cell);
}


template<int dim, bool is_multigrid>
std::pair<std::vector<std::array<unsigned int, 2>>, std::vector<types::global_dof_index>>
MatrixIntegrator<dim, is_multigrid>::make_joint_interface_indices(
  const std::vector<unsigned int> &            testfunc_indices_left,
  const std::vector<types::global_dof_index> & dof_indices_on_lcell,
  const std::vector<unsigned int> &            testfunc_indices_right,
  const std::vector<types::global_dof_index> & dof_indices_on_rcell) const
{
  AssertDimension(testfunc_indices_left.size(), dof_indices_on_lcell.size());
  AssertDimension(testfunc_indices_right.size(), dof_indices_on_rcell.size());

  std::vector<std::pair<unsigned int, types::global_dof_index>> testfunc_and_dof_indices_left;
  std::transform(testfunc_indices_left.cbegin(),
                 testfunc_indices_left.cend(),
                 dof_indices_on_lcell.cbegin(),
                 std::back_inserter(testfunc_and_dof_indices_left),
                 [](const auto i,
                    const auto dof) -> std::pair<unsigned int, types::global_dof_index> {
                   return {i, dof};
                 });
  std::vector<std::pair<unsigned int, types::global_dof_index>> testfunc_and_dof_indices_right;
  std::transform(testfunc_indices_right.cbegin(),
                 testfunc_indices_right.cend(),
                 dof_indices_on_rcell.cbegin(),
                 std::back_inserter(testfunc_and_dof_indices_right),
                 [](const auto i,
                    const auto dof) -> std::pair<unsigned int, types::global_dof_index> {
                   return {i, dof};
                 });

  std::pair<std::vector<std::array<unsigned int, 2>>, std::vector<types::global_dof_index>> indices;
  auto & [joint_testfunc_indices, interface_dof_indices] = indices;

  for(const auto [li, ldof] : testfunc_and_dof_indices_left)
  {
    interface_dof_indices.push_back(ldof);
    joint_testfunc_indices.push_back({li, numbers::invalid_unsigned_int});
  }

  for(const auto [ri, rdof] : testfunc_and_dof_indices_right)
  {
    const auto common_it =
      std::find_if(testfunc_and_dof_indices_left.cbegin(),
                   testfunc_and_dof_indices_left.cend(),
                   [&](const auto & li_and_ldof) { return li_and_ldof.second == rdof; });
    const bool is_common_dof = common_it != testfunc_and_dof_indices_left.cend();

    if(!is_common_dof)
    {
      interface_dof_indices.push_back(rdof);
      joint_testfunc_indices.push_back({numbers::invalid_unsigned_int, ri});
    }

    else
    {
      const auto [li, ldof] = *common_it;
      (void)ldof;
      auto joint_it = std::find_if(joint_testfunc_indices.begin(),
                                   joint_testfunc_indices.end(),
                                   [li](const auto li_and_ri) { return li_and_ri[0] == li; });
      Assert(joint_it != joint_testfunc_indices.end(), ExcMessage("..."));
      (*joint_it)[1] = ri;
    }
  }

  AssertDimension(joint_testfunc_indices.size(), interface_dof_indices.size());
  return indices;
}

} // namespace MW



namespace FD
{
template<int dim,
         int fe_degree,
         typename Number            = double,
         TPSS::DoFLayout dof_layout = TPSS::DoFLayout::Q>
class MatrixIntegrator
{
public:
  using This = MatrixIntegrator<dim, fe_degree, Number>;

  static constexpr int n_q_points_1d = fe_degree + 1 + (dof_layout == TPSS::DoFLayout::RT ? 1 : 0);

  using value_type     = Number;
  using transfer_type  = typename TPSS::PatchTransfer<dim, Number>;
  using matrix_type_1d = Table<2, VectorizedArray<Number>>;
  using matrix_type    = Tensors::BlockMatrix<dim, VectorizedArray<Number>, -1, -1>;
  using evaluator_type = FDEvaluation<dim, fe_degree, n_q_points_1d, Number>;

  void
  initialize(const EquationData & equation_data_in)
  {
    equation_data = equation_data_in;
  }

  template<typename OperatorType>
  void
  assemble_subspace_inverses(const SubdomainHandler<dim, Number> & subdomain_handler,
                             std::vector<matrix_type> &            local_matrices,
                             const OperatorType &,
                             const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    AssertDimension(subdomain_handler.get_partition_data().n_subdomains(), local_matrices.size());
    /// TODO tangential components only for RT !!!

    constexpr bool is_sipg =
      TPSS::DoFLayout::DGQ == dof_layout || TPSS::DoFLayout::RT == dof_layout;

    // const auto zero_out = [](std::vector<std::array<matrix_type_1d, dim>> & rank1_tensors) {
    //   for(auto & tensor : rank1_tensors)
    //   {
    //     tensor.front() = 0. * tensor.front();
    //   }
    // };

    for(auto comp_test = 0U; comp_test < dim; ++comp_test)
    {
      evaluator_type eval_test(subdomain_handler, /*dofh_index*/ 0, comp_test);
      for(auto comp_ansatz = comp_test; comp_ansatz < dim; ++comp_ansatz) // assuming isotropy !
      {
        evaluator_type eval_ansatz(subdomain_handler, /*dofh_index*/ 0, comp_ansatz);
        for(unsigned int patch = subdomain_range.first; patch < subdomain_range.second; ++patch)
        {
          auto & velocity_matrix = local_matrices[patch];
          if(velocity_matrix.n_block_rows() != dim && velocity_matrix.n_block_cols() != dim)
            velocity_matrix.resize(dim, dim);

          eval_test.reinit(patch);
          eval_ansatz.reinit(patch);

          const auto mass_matrices = assemble_mass_tensor(eval_test, eval_ansatz);

          if(comp_test == comp_ansatz)
          {
            const auto laplace_matrices = assemble_laplace_tensor<is_sipg>(eval_test, eval_ansatz);

            const auto & MxMxL = [&](const unsigned int direction_of_L) {
              /// For example, we obtain MxMxL for direction_of_L = 0 (dimension
              /// 0 is rightmost!)
              std::array<matrix_type_1d, dim> kronecker_tensor;
              /// if direction_of_L equals the velocity component we scale by two
              AssertDimension(comp_test, comp_ansatz);
              const auto factor = direction_of_L == comp_ansatz ? 2. : 1.;
              for(auto d = 0U; d < dim; ++d)
                kronecker_tensor[d] = d == direction_of_L ?
                                        factor * laplace_matrices[direction_of_L] :
                                        mass_matrices[d];
              return kronecker_tensor;
            };

            /// (0,0)-block: LxMxM + MxLxM + MxMx2L
            std::vector<std::array<matrix_type_1d, dim>> rank1_tensors;
            for(auto direction_of_L = 0; direction_of_L < dim; ++direction_of_L)
              rank1_tensors.emplace_back(MxMxL(direction_of_L));
            velocity_matrix.get_block(comp_test, comp_ansatz).reinit(rank1_tensors);
          }

          else
          {
            /// The factor 2 arising from 2 * e(u) : grad v is implicitly
            /// equalized by the factor 1/2 from the symmetrized gradient
            /// e(u). First, we emphasize that for off-diagonal blocks there
            /// are no penalty contributions. For the remaing contributions,
            /// namely consistency and symmetry terms, again the factor 2 is
            /// implicitly equalized. Nevertheless, we have to consider the
            /// factor 1/2 arising from average operators {{e(u)}} and
            /// {{e(v)}}, respectively.
            const auto gradient_matrices = assemble_gradient_tensor(eval_test, eval_ansatz);

            const auto & MxGxGT = [&](const auto component_test, const auto component_ansatz) {
              const int deriv_index_ansatz = component_test;
              const int deriv_index_test   = component_ansatz;
              Assert(deriv_index_ansatz != deriv_index_test,
                     ExcMessage("This case is not well-defined."));
              std::array<matrix_type_1d, dim> kronecker_tensor;
              for(auto d = 0; d < dim; ++d)
              {
                if(d == deriv_index_ansatz)
                  kronecker_tensor[d] = gradient_matrices[deriv_index_ansatz];
                else if(d == deriv_index_test)
                  kronecker_tensor[d] = LinAlg::transpose(gradient_matrices[deriv_index_test]);
                else
                  kronecker_tensor[d] = mass_matrices[d];
              }
              return kronecker_tensor;
            };

            std::vector<std::array<matrix_type_1d, dim>> rank1_tensors;
            /// (0,1)-block: MxGxGT + MxPxG + MxGTxP
            {
              /// MxGxGT
              rank1_tensors.emplace_back(MxGxGT(comp_test, comp_ansatz));

              /// Factor 1/2 of average operator {{e(u)}} and {{e(v)}} is used
              /// within assemble_mixed_nitsche_tensor
              const auto point_mass_matrices =
                assemble_mixed_nitsche_tensor(eval_test, eval_ansatz);

              { /// MxPxG
                std::array<matrix_type_1d, dim> kronecker_tensor;
                for(auto d = 0U; d < dim; ++d)
                {
                  if(d == comp_test)
                    kronecker_tensor[d] = gradient_matrices[comp_test];
                  else if(d == comp_ansatz)
                    kronecker_tensor[d] = point_mass_matrices[comp_ansatz];
                  else
                    kronecker_tensor[d] = mass_matrices[d];
                }
                rank1_tensors.emplace_back(kronecker_tensor);
              }

              { /// MxGTxP
                std::array<matrix_type_1d, dim> kronecker_tensor;
                for(auto d = 0U; d < dim; ++d)
                {
                  if(d == comp_test)
                    kronecker_tensor[d] = point_mass_matrices[comp_test];
                  else if(d == comp_ansatz)
                    kronecker_tensor[d] = LinAlg::transpose(gradient_matrices[comp_ansatz]);
                  else
                    kronecker_tensor[d] = mass_matrices[d];
                }
                rank1_tensors.emplace_back(kronecker_tensor);
              }

              velocity_matrix.get_block(comp_test, comp_ansatz).reinit(rank1_tensors);
            }

            /// (1,0)-block: transpose of (0,1)-block
            {
              for(auto & tensor : rank1_tensors)
                Tensors::transpose_tensor<dim>(tensor);
              velocity_matrix.get_block(comp_ansatz, comp_test).reinit(rank1_tensors);
            }
          }
        }
      }
    }
  }

  std::array<matrix_type_1d, dim>
  assemble_mixed_nitsche_tensor(evaluator_type & eval_test, evaluator_type & eval_ansatz) const
  {
    using CellVoid = ::FD::Void::CellOperation<dim, fe_degree, n_q_points_1d, Number>;

    const auto face_point_mass = [&](const evaluator_type &              eval_ansatz,
                                     const evaluator_type &              eval_test,
                                     Table<2, VectorizedArray<Number>> & cell_matrix,
                                     const int                           direction,
                                     const int                           cell_no,
                                     const int                           face_no) {
      const int  n_dofs_test    = eval_test.n_dofs_per_cell_1d(direction);
      const int  n_dofs_ansatz  = eval_ansatz.n_dofs_per_cell_1d(direction);
      const auto normal_vector  = eval_ansatz.get_normal_vector(face_no, direction);
      const auto average_factor = eval_test.get_average_factor(direction, cell_no, face_no);
      const auto comp_u         = eval_ansatz.vector_component();
      const auto comp_v         = eval_test.vector_component();

      for(int i = 0; i < n_dofs_test; ++i)
      {
        const auto & v_i = eval_test.shape_value_face(i, face_no, direction, cell_no);
        for(int j = 0; j < n_dofs_ansatz; ++j)
        {
          const auto & u_j           = eval_ansatz.shape_value_face(j, face_no, direction, cell_no);
          const auto & value_on_face = -average_factor * (v_i * normal_vector[comp_u] * u_j +
                                                          v_i * u_j * normal_vector[comp_v]);
          cell_matrix(i, j) += value_on_face;
        }
      }
    };

    const auto interface_point_mass = [&](const evaluator_type &              eval_ansatz,
                                          const evaluator_type &              eval_test,
                                          Table<2, VectorizedArray<Number>> & cell_matrix01,
                                          Table<2, VectorizedArray<Number>> & cell_matrix10,
                                          const int                           cell_no_left,
                                          const int                           direction) {
      (void)cell_no_left;
      AssertDimension(cell_no_left, 0);
      const int  n_dofs_test    = eval_test.n_dofs_per_cell_1d(direction);
      const int  n_dofs_ansatz  = eval_ansatz.n_dofs_per_cell_1d(direction);
      const auto normal_vector0 = eval_test.get_normal_vector(1, direction); // on cell 0
      const auto normal_vector1 = eval_test.get_normal_vector(0, direction); // on cell 1
      const auto comp_u         = eval_ansatz.vector_component();
      const auto comp_v         = eval_test.vector_component();

      auto value_on_interface01{make_vectorized_array<Number>(0.)};
      auto value_on_interface10{make_vectorized_array<Number>(0.)};
      for(int i = 0; i < n_dofs_test; ++i)
      {
        const auto & v0_i = eval_test.shape_value_face(i, /*face_no*/ 1, direction, /*cell_no*/ 0);
        const auto & v1_i = eval_test.shape_value_face(i, 0, direction, 1);
        for(int j = 0; j < n_dofs_ansatz; ++j)
        {
          const auto & u0_j = eval_ansatz.shape_value_face(j, 1, direction, 0);
          const auto & u1_j = eval_ansatz.shape_value_face(j, 0, direction, 1);

          /// consistency + symmetry
          value_on_interface01 =
            -0.5 * (v0_i * normal_vector0[comp_u] * u1_j + v0_i * u1_j * normal_vector1[comp_v]);
          value_on_interface10 =
            -0.5 * (v1_i * normal_vector1[comp_u] * u0_j + v1_i * u0_j * normal_vector0[comp_v]);

          cell_matrix01(i, j) += value_on_interface01;
          cell_matrix10(i, j) += value_on_interface10;
        }
      }
    };

    return eval_test.patch_action(eval_ansatz, CellVoid{}, face_point_mass, interface_point_mass);
  }

  template<bool is_sipg = false>
  std::array<matrix_type_1d, dim>
  assemble_laplace_tensor(evaluator_type & eval_test, evaluator_type & eval_ansatz) const
  {
    using CellLaplace = ::FD::Laplace::CellOperation<dim, fe_degree, n_q_points_1d, Number>;
    CellLaplace cell_laplace;

    if constexpr(is_sipg)
    {
      using FaceLaplace = ::FD::Laplace::SIPG::FaceOperation<dim, fe_degree, n_q_points_1d, Number>;
      FaceLaplace nitsche;
      nitsche.penalty_factor          = equation_data.ip_factor;
      nitsche.interior_penalty_factor = equation_data.ip_factor;

      const auto face_nitsche_plus_penalty = [&](const evaluator_type & eval_ansatz,
                                                 const evaluator_type & eval_test,
                                                 matrix_type_1d &       cell_matrix,
                                                 const int              direction,
                                                 const int              cell_no,
                                                 const int              face_no) {
        nitsche(eval_ansatz, eval_test, cell_matrix, direction, cell_no, face_no);

        const int vector_component = eval_test.vector_component();
        AssertDimension(vector_component, static_cast<int>(eval_ansatz.vector_component()));

        if(vector_component != direction)
        {
          const int  n_dofs_test    = eval_test.n_dofs_per_cell_1d(direction);
          const int  n_dofs_ansatz  = eval_ansatz.n_dofs_per_cell_1d(direction);
          const auto average_factor = eval_test.get_average_factor(direction, cell_no, face_no);
          const auto normal         = eval_test.get_normal(face_no);

          const auto h       = eval_test.get_h(direction, cell_no);
          const auto penalty = nitsche.penalty_factor * average_factor *
                               ::Nitsche::compute_penalty_impl(fe_degree, h, h);

          auto value_on_face = make_vectorized_array<Number>(0.);
          for(int i = 0; i < n_dofs_test; ++i)
          {
            const auto & v_i = eval_test.shape_value_face(i, face_no, direction, cell_no);
            for(int j = 0; j < n_dofs_ansatz; ++j)
            {
              const auto & u_j = eval_ansatz.shape_value_face(j, face_no, direction, cell_no);
              value_on_face    = penalty * v_i * u_j * normal * normal;
              cell_matrix(i, j) += value_on_face;
            }
          }
        }
      };

      const auto interface_nitsche_plus_penalty = [&](const evaluator_type & eval_ansatz,
                                                      const evaluator_type & eval_test,
                                                      matrix_type_1d &       cell_matrix01,
                                                      matrix_type_1d &       cell_matrix10,
                                                      const int              cell_no0,
                                                      const int              direction) {
        nitsche(eval_ansatz, eval_test, cell_matrix01, cell_matrix10, cell_no0, direction);

        const int vector_component = eval_test.vector_component();
        AssertDimension(vector_component, static_cast<int>(eval_ansatz.vector_component()));

        if(vector_component != direction)
        {
          const int  n_dofs_test   = eval_test.n_dofs_per_cell_1d(direction);
          const int  n_dofs_ansatz = eval_ansatz.n_dofs_per_cell_1d(direction);
          const auto normal0       = eval_test.get_normal(1); // on cell 0
          const auto normal1       = eval_test.get_normal(0); // on cell 1

          const auto h0      = eval_test.get_h(direction, cell_no0);
          const auto h1      = eval_test.get_h(direction, cell_no0 + 1);
          const auto penalty = nitsche.interior_penalty_factor * 0.5 *
                               ::Nitsche::compute_penalty_impl(fe_degree, h0, h1);

          auto value_on_interface01 = make_vectorized_array<Number>(0.);
          auto value_on_interface10 = make_vectorized_array<Number>(0.);
          for(int i = 0; i < n_dofs_test; ++i)
          {
            const auto & v0_i = eval_test.shape_value_face(i, 1, direction, 0);
            const auto & v1_i = eval_test.shape_value_face(i, 0, direction, 1);
            for(int j = 0; j < n_dofs_ansatz; ++j)
            {
              const auto & u0_j    = eval_ansatz.shape_value_face(j, 1, direction, 0);
              const auto & u1_j    = eval_ansatz.shape_value_face(j, 0, direction, 1);
              value_on_interface01 = penalty * v0_i * u1_j * normal0 * normal1;
              value_on_interface10 = penalty * v1_i * u0_j * normal1 * normal0;
              cell_matrix01(i, j) += value_on_interface01;
              cell_matrix10(i, j) += value_on_interface10;
            }
          }
        }
      };

      return eval_test.patch_action(eval_ansatz,
                                    cell_laplace,
                                    face_nitsche_plus_penalty,
                                    interface_nitsche_plus_penalty);
    }

    return eval_test.patch_action(eval_ansatz, cell_laplace);
  }

  std::array<matrix_type_1d, dim>
  assemble_mass_tensor(evaluator_type & eval_test, evaluator_type & eval_ansatz) const
  {
    using CellMass = ::FD::L2::CellOperation<dim, fe_degree, n_q_points_1d, Number>;
    return eval_test.patch_action(eval_ansatz, CellMass{});
  }

  std::array<matrix_type_1d, dim>
  assemble_gradient_tensor(evaluator_type & eval_test, evaluator_type & eval_ansatz) const
  {
    using CellGradient = ::FD::Gradient::CellOperation<dim, fe_degree, n_q_points_1d, Number>;
    CellGradient cell_gradient;
    return eval_test.patch_action(eval_ansatz, cell_gradient);
  }

  std::shared_ptr<transfer_type>
  get_patch_transfer(const SubdomainHandler<dim, Number> & subdomain_handler) const
  {
    return std::make_shared<transfer_type>(subdomain_handler);
  }

  EquationData equation_data;
};

} // end namespace FD

} // end namespace SIPG

} // namespace Velocity



namespace Pressure
{
namespace MW
{
using ::MW::ScratchData;

using ::MW::DoF::CopyData;

template<int dim, bool is_multigrid = false>
struct MatrixIntegrator
{
  using IteratorType = typename ::MW::IteratorSelector<dim, is_multigrid>::type;

  MatrixIntegrator(const Function<dim> *                              load_function_in,
                   const Function<dim> *                              analytical_solution_in,
                   const LinearAlgebra::distributed::Vector<double> * particular_solution,
                   const EquationData &                               equation_data_in)
    : load_function(load_function_in),
      analytical_solution(analytical_solution_in),
      discrete_solution(particular_solution),
      equation_data(equation_data_in)
  {
  }

  void
  cell_worker(const IteratorType & cell,
              ScratchData<dim> &   scratch_data,
              CopyData &           copy_data) const;

  void
  cell_mass_worker(const IteratorType & cell,
                   ScratchData<dim> &   scratch_data,
                   CopyData &           copy_data) const;

  const Function<dim> *                              load_function;
  const Function<dim> *                              analytical_solution;
  const LinearAlgebra::distributed::Vector<double> * discrete_solution;
  const EquationData                                 equation_data;
};

template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::cell_worker(const IteratorType & cell,
                                                 ScratchData<dim> &   scratch_data,
                                                 CopyData &           copy_data) const
{
  AssertDimension(copy_data.cell_data.size(), 0U);

  FEValues<dim> & phi = scratch_data.fe_values;
  phi.reinit(cell);

  const unsigned int dofs_per_cell = phi.get_fe().dofs_per_cell;

  auto & cell_data = copy_data.cell_data.emplace_back(dofs_per_cell);

  cell->get_active_or_mg_dof_indices(cell_data.dof_indices);

  AssertDimension(load_function->n_components, 1U);

  const auto & quadrature_points = phi.get_quadrature_points();
  for(unsigned int q = 0; q < phi.n_quadrature_points; ++q)
  {
    const auto & x_q = quadrature_points[q];
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      if(!is_multigrid)
      {
        const auto load_value = load_function->value(x_q);
        cell_data.rhs(i) += phi.shape_value(i, q) * load_value * phi.JxW(q);
      }
    }
  }
}

template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::cell_mass_worker(const IteratorType & cell,
                                                      ScratchData<dim> &   scratch_data,
                                                      CopyData &           copy_data) const
{
  AssertDimension(copy_data.cell_data.size(), 0U);

  FEValues<dim> & phi = scratch_data.fe_values;
  phi.reinit(cell);

  const unsigned int dofs_per_cell = phi.get_fe().dofs_per_cell;

  auto & cell_data = copy_data.cell_data.emplace_back(dofs_per_cell);

  cell->get_active_or_mg_dof_indices(cell_data.dof_indices);

  for(unsigned int q = 0; q < phi.n_quadrature_points; ++q)
    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        cell_data.matrix(i, j) += phi.shape_value(i, q) * phi.shape_value(j, q) * phi.JxW(q);
}

} // namespace MW

} // namespace Pressure



namespace VelocityPressure
{
namespace MW
{
using ::MW::ScratchData;

using ::MW::CopyData;



template<int dim, bool is_multigrid = false>
struct MatrixIntegrator
{
  static_assert(!is_multigrid, "not implemented.");

  using IteratorType = typename ::MW::IteratorSelector<dim, is_multigrid>::type;

  MatrixIntegrator(const Function<dim> *                                   load_function_in,
                   const Function<dim> *                                   analytical_solution_in,
                   const LinearAlgebra::distributed::BlockVector<double> * particular_solution,
                   const EquationData &                                    equation_data_in)
    : load_function(load_function_in),
      analytical_solution(analytical_solution_in),
      discrete_solution(particular_solution),
      equation_data(equation_data_in)
  {
  }

  void
  cell_worker(const IteratorType & cell,
              ScratchData<dim> &   scratch_data,
              CopyData &           copy_data) const
  {
    copy_data.cell_matrix = 0.;
    copy_data.cell_rhs    = 0.;

    FEValues<dim> & fe_values = scratch_data.fe_values;
    fe_values.reinit(cell);
    cell->get_active_or_mg_dof_indices(copy_data.local_dof_indices);
    const auto &       fe            = fe_values.get_fe();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = fe_values.n_quadrature_points;

    std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim + 1));

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
    std::vector<double>                  div_phi_u(dofs_per_cell);
    std::vector<double>                  phi_p(dofs_per_cell);

    load_function->vector_value_list(fe_values.get_quadrature_points(), rhs_values);

    for(unsigned int q = 0; q < n_q_points; ++q)
    {
      for(unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient(k, q);
        div_phi_u[k]     = fe_values[velocities].divergence(k, q);
        phi_p[k]         = fe_values[pressure].value(k, q);
      }

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for(unsigned int j = 0; j <= i; ++j)
        {
          copy_data.cell_matrix(i, j) +=
            (2 * (symgrad_phi_u[i] * symgrad_phi_u[j]) - div_phi_u[i] * phi_p[j] -
             phi_p[i] * div_phi_u[j] +
             (equation_data.assemble_pressure_mass_matrix ? phi_p[i] * phi_p[j] : 0)) *
            fe_values.JxW(q);
        }

        const unsigned int component_i = fe.system_to_component_index(i).first;
        copy_data.cell_rhs(i) +=
          fe_values.shape_value(i, q) * rhs_values[q](component_i) * fe_values.JxW(q);
      }
    }

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      for(unsigned int j = i + 1; j < dofs_per_cell; ++j)
        copy_data.cell_matrix(i, j) = copy_data.cell_matrix(j, i);

    if(discrete_solution)
    {
      Vector<double> u0(copy_data.local_dof_indices.size());
      for(auto i = 0U; i < u0.size(); ++i)
        u0(i) = (*discrete_solution)(copy_data.local_dof_indices[i]);
      Vector<double> w0(copy_data.local_dof_indices.size());
      copy_data.cell_matrix.vmult(w0, u0);
      copy_data.cell_rhs -= w0;
    }
  }

  const Function<dim> *                                   load_function;
  const Function<dim> *                                   analytical_solution;
  const LinearAlgebra::distributed::BlockVector<double> * discrete_solution;
  const EquationData                                      equation_data;
};



namespace Mixed
{
using ::MW::compute_vvalue;

using ::MW::compute_vjump;

using ::MW::compute_vjump_dot_normal;

using ::MW::compute_divergence;

using ::MW::Mixed::ScratchData;

using CopyData = std::array<::MW::DoF::CopyData, 2>;

template<int dim, bool is_multigrid = false>
struct MatrixIntegrator
{
  using IteratorType = typename ::MW::IteratorSelector<dim, is_multigrid>::type;

  MatrixIntegrator(const LinearAlgebra::distributed::Vector<double> * particular_solutionU,
                   const LinearAlgebra::distributed::Vector<double> * particular_solutionP,
                   const Function<dim> *                              analytical_solutionU_in,
                   const Function<dim> *                              analytical_solutionP_in,
                   const EquationData &                               equation_data_in)
    : discrete_solutionU(particular_solutionU),
      discrete_solutionP(particular_solutionP),
      analytical_solutionU(analytical_solutionU_in),
      analytical_solutionP(analytical_solutionP_in),
      equation_data(equation_data_in)
  {
    AssertThrow(
      !particular_solutionP,
      ExcMessage(
        "There is currently no reason to set particular_solutionP, as it should be a vector filled with zeros!"));
    AssertThrow(
      !analytical_solutionP,
      ExcMessage(
        "There is currently no reason to set analytical_solutionP, as it is not used by the bilinear form!"));
  }

  void
  cell_worker(const IteratorType & cellU,
              const IteratorType & cellP,
              ScratchData<dim> &   scratch_data,
              CopyData &           copy_data) const;

  template<typename TestEvaluatorType, typename AnsatzEvaluatorType>
  void
  cell_worker_impl(const TestEvaluatorType &       phi_test,
                   const AnsatzEvaluatorType &     phi_ansatz,
                   ::MW::DoF::CopyData::CellData & cell_data) const;

  void
  face_worker(const IteratorType & cellU,
              const IteratorType & cellP,
              const unsigned int & f,
              const unsigned int & sf,
              const IteratorType & ncellU,
              const IteratorType & ncellP,
              const unsigned int & nf,
              const unsigned int & nsf,
              ScratchData<dim> &   scratch_data,
              CopyData &           copy_data) const;

  template<bool is_uniface>
  void
  boundary_or_uniface_worker(const IteratorType & cellU,
                             const IteratorType & cellP,
                             const unsigned int & f,
                             const unsigned int & sf,
                             ScratchData<dim> &   scratch_data,
                             CopyData &           copy_data) const;

  void
  uniface_worker(const IteratorType & cellU,
                 const IteratorType & cellP,
                 const unsigned int & f,
                 const unsigned int & sf,
                 ScratchData<dim> &   scratch_data,
                 CopyData &           copy_data) const;

  void
  boundary_worker(const IteratorType & cellU,
                  const IteratorType & cellP,
                  const unsigned int & f,
                  ScratchData<dim> &   scratch_data,
                  CopyData &           copy_data) const;

  const LinearAlgebra::distributed::Vector<double> * discrete_solutionU;
  const LinearAlgebra::distributed::Vector<double> * discrete_solutionP;
  const Function<dim> *                              analytical_solutionU;
  const Function<dim> *                              analytical_solutionP;
  const EquationData                                 equation_data;
};


template<int dim, bool is_multigrid>
template<typename TestEvaluatorType, typename AnsatzEvaluatorType>
void
MatrixIntegrator<dim, is_multigrid>::cell_worker_impl(
  const TestEvaluatorType &       phiU,
  const AnsatzEvaluatorType &     phiP,
  ::MW::DoF::CopyData::CellData & cell_data) const
{
  AssertDimension(phiU.n_quadrature_points, phiP.n_quadrature_points);

  for(unsigned int q = 0; q < phiU.n_quadrature_points; ++q)
  {
    for(unsigned int i = 0; i < cell_data.dof_indices.size(); ++i) // test
    {
      const auto div_phiU_i = compute_divergence(phiU, i, q);
      for(unsigned int j = 0; j < cell_data.dof_indices_column.size(); ++j) // ansatz
      {
        const auto phiP_j = phiP.shape_value(j, q);

        cell_data.matrix(i, j) += -div_phiU_i * phiP_j * phiU.JxW(q);
      }
    }
  }
}


template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::cell_worker(const IteratorType & cellU,
                                                 const IteratorType & cellP,
                                                 ScratchData<dim> &   scratch_data,
                                                 CopyData &           copy_data_pair) const
{
  auto & [copy_data, copy_data_flipped] = copy_data_pair;

  AssertDimension(copy_data.cell_data.size(), 0U);
  AssertDimension(copy_data_flipped.cell_data.size(), 0U);
  AssertDimension(cellU->index(), cellP->index());

  /// Velocity "U" takes test function role (in flipped mode ansatz function)
  auto & phiU = scratch_data.fe_values_test;
  phiU.reinit(cellU);
  const auto n_dofs_per_cellU = phiU.dofs_per_cell;

  /// Pressure "P" takes ansatz function role (in flipped mode test function)
  auto & phiP = scratch_data.fe_values_ansatz;
  phiP.reinit(cellP);
  const auto n_dofs_per_cellP = phiP.dofs_per_cell;

  auto & cell_data = copy_data.cell_data.emplace_back(n_dofs_per_cellU, n_dofs_per_cellP);
  auto & cell_data_flipped =
    copy_data_flipped.cell_data.emplace_back(n_dofs_per_cellP, n_dofs_per_cellU);

  cellU->get_active_or_mg_dof_indices(cell_data.dof_indices);
  cellP->get_active_or_mg_dof_indices(cell_data.dof_indices_column);

  cell_worker_impl(phiU, phiP, cell_data);

  /// pressure-velocity block ("flipped") is the transpose of the
  /// velocity-pressure block
  for(unsigned int i = 0; i < n_dofs_per_cellU; ++i)
    for(unsigned int j = 0; j < n_dofs_per_cellP; ++j)
      cell_data_flipped.matrix(j, i) = cell_data.matrix(i, j);

  /// Lifting of inhomogeneous boundary condition
  if(!is_multigrid)
    if(discrete_solutionU && cellU->at_boundary())
    {
      AssertDimension(n_dofs_per_cellU, cell_data.dof_indices.size());
      AssertDimension(n_dofs_per_cellP, cell_data.dof_indices_column.size());
      Vector<double> u0(n_dofs_per_cellU);
      for(auto i = 0U; i < u0.size(); ++i)
        u0(i) = (*discrete_solutionU)(cell_data.dof_indices[i]);
      Vector<double> w0(n_dofs_per_cellP);
      cell_data_flipped.matrix.vmult(w0, u0);
      AssertDimension(n_dofs_per_cellP, cell_data_flipped.rhs.size());
      cell_data_flipped.rhs -= w0;
    }
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::face_worker(const IteratorType & cellU,
                                                 const IteratorType & cellP,
                                                 const unsigned int & f,
                                                 const unsigned int & sf,
                                                 const IteratorType & ncellU,
                                                 const IteratorType & ncellP,
                                                 const unsigned int & nf,
                                                 const unsigned int & nsf,
                                                 ScratchData<dim> &   scratch_data,
                                                 CopyData &           copy_data_pair) const
{
  auto & [copy_data, copy_data_flipped] = copy_data_pair;

  /// Velocity "U" takes test function role (in flipped mode ansatz function)
  auto & phiU = scratch_data.fe_interface_values_test;
  phiU.reinit(cellU, f, sf, ncellU, nf, nsf);
  const auto n_dofsU = phiU.n_current_interface_dofs();

  /// Pressure "P" takes ansatz function role (in flipped mode test function)
  auto & phiP = scratch_data.fe_interface_values_ansatz;
  phiP.reinit(cellP, f, sf, ncellP, nf, nsf);
  const auto n_dofsP = phiP.n_current_interface_dofs();

  AssertDimension(phiU.n_quadrature_points, phiP.n_quadrature_points);

  auto & face_data         = copy_data.face_data.emplace_back(n_dofsU, n_dofsP);
  auto & face_data_flipped = copy_data_flipped.face_data.emplace_back(n_dofsP, n_dofsU);

  face_data.dof_indices        = phiU.get_interface_dof_indices();
  face_data.dof_indices_column = phiP.get_interface_dof_indices();

  AssertDimension(n_dofsU, face_data.dof_indices.size());
  AssertDimension(n_dofsP, face_data.dof_indices_column.size());

  double integral_ijq = 0.;
  for(unsigned int q = 0; q < phiU.n_quadrature_points; ++q)
  {
    for(unsigned int i = 0; i < n_dofsU; ++i)
    {
      const auto & jump_phiU_i_dot_n = compute_vjump_dot_normal(phiU, i, q);

      for(unsigned int j = 0; j < n_dofsP; ++j)
      {
        const auto & av_phiP_j = phiP.average(j, q);

        integral_ijq = av_phiP_j * jump_phiU_i_dot_n * phiU.JxW(q);

        face_data.matrix(i, j) += integral_ijq;
      }
    }
  }

  /// pressure-velocity block ("flipped") is the transpose of the
  /// velocity-pressure block
  face_data_flipped.matrix.reinit(n_dofsP, n_dofsU);
  for(unsigned int i = 0; i < n_dofsU; ++i)
    for(unsigned int j = 0; j < n_dofsP; ++j)
      face_data_flipped.matrix(j, i) = face_data.matrix(i, j);
}



template<int dim, bool is_multigrid>
template<bool is_uniface>
void
MatrixIntegrator<dim, is_multigrid>::boundary_or_uniface_worker(const IteratorType & cellU,
                                                                const IteratorType & cellP,
                                                                const unsigned int & f,
                                                                const unsigned int & sf,
                                                                ScratchData<dim> &   scratch_data,
                                                                CopyData & copy_data_pair) const
{
  constexpr bool do_rhs = !is_multigrid && !is_uniface;

  if(!is_uniface)
    AssertDimension(sf, numbers::invalid_unsigned_int); // prevent from being used

  auto & [copy_data, copy_data_flipped] = copy_data_pair;

  /// Velocity "U" takes test function role (in flipped mode ansatz function)
  if(is_uniface)
    scratch_data.fe_interface_values_test.reinit(cellU, f, sf, cellU, f, sf);
  else
    scratch_data.fe_interface_values_test.reinit(cellU, f);
  const auto & phiU    = scratch_data.fe_interface_values_test.get_fe_face_values(0);
  const auto   n_dofsU = phiU.dofs_per_cell; // phiU.n_current_interface_dofs();

  /// Pressure "P" takes ansatz function role (in flipped mode test function)
  if(is_uniface)
    scratch_data.fe_interface_values_ansatz.reinit(cellP, f, sf, cellP, f, sf);
  else
    scratch_data.fe_interface_values_ansatz.reinit(cellP, f);
  const auto & phiP    = scratch_data.fe_interface_values_ansatz.get_fe_face_values(0);
  const auto   n_dofsP = phiP.dofs_per_cell;

  auto & face_data         = copy_data.face_data.emplace_back(n_dofsU, n_dofsP);
  auto & face_data_flipped = copy_data_flipped.face_data.emplace_back(n_dofsP, n_dofsU);

  cellU->get_active_or_mg_dof_indices(face_data.dof_indices);
  cellP->get_active_or_mg_dof_indices(face_data.dof_indices_column);

  const std::vector<Tensor<1, dim>> & normals = phiU.get_normal_vectors();

  AssertDimension(phiU.n_quadrature_points, phiP.n_quadrature_points);
  std::vector<double> velocity_solution_dot_normals;
  if(do_rhs)
  {
    Assert(analytical_solutionU, ExcMessage("analytical_solutionU is not set."));
    AssertDimension(analytical_solutionU->n_components, dim);
    const auto & q_points = phiU.get_quadrature_points();
    std::transform(q_points.cbegin(),
                   q_points.cend(),
                   normals.cbegin(),
                   std::back_inserter(velocity_solution_dot_normals),
                   [this](const auto & x_q, const auto & normal) {
                     Tensor<1, dim> u_q;
                     for(auto c = 0U; c < dim; ++c)
                       u_q[c] = analytical_solutionU->value(x_q, c);
                     return u_q * normal;
                   });
  }

  double integral_ijq = 0.;
  double integral_jq  = 0.;
  for(unsigned int q = 0; q < phiP.n_quadrature_points; ++q)
  {
    for(unsigned int j = 0; j < n_dofsP; ++j)
    {
      const auto & av_phiP_j = is_uniface ? 0.5 * phiP.shape_value(j, q) : phiP.shape_value(j, q);

      /// weak Dirichlet conditions (P is test function)
      if(do_rhs)
      {
        const auto & u_dot_n =
          velocity_solution_dot_normals[q]; // !!! should be zero for hdiv conf method

        integral_jq = u_dot_n * av_phiP_j * phiP.JxW(q);

        face_data_flipped.rhs(j) += integral_jq;
      }

      /// interior penalty contribution
      for(unsigned int i = 0; i < n_dofsU; ++i)
      {
        const auto & jump_phiU         = compute_vvalue(phiU, i, q);
        const auto & n                 = normals[q];
        const auto & jump_phiU_i_dot_n = jump_phiU * n;

        integral_ijq = av_phiP_j * jump_phiU_i_dot_n * phiU.JxW(q);

        face_data.matrix(i, j) += integral_ijq;
      }
    }
  }

  /// The pressure-velocity block ("flipped") is the transpose of the
  /// velocity-pressure block.
  for(unsigned int i = 0; i < n_dofsU; ++i)
    for(unsigned int j = 0; j < n_dofsP; ++j)
      face_data_flipped.matrix(j, i) = face_data.matrix(i, j);

  AssertDimension(copy_data.face_data.size(), copy_data_flipped.face_data.size());
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::uniface_worker(const IteratorType & cellU,
                                                    const IteratorType & cellP,
                                                    const unsigned int & f,
                                                    const unsigned int & sf,
                                                    ScratchData<dim> &   scratch_data,
                                                    CopyData &           copy_data_pair) const
{
  boundary_or_uniface_worker<true>(cellU, cellP, f, sf, scratch_data, copy_data_pair);
}



template<int dim, bool is_multigrid>
void
MatrixIntegrator<dim, is_multigrid>::boundary_worker(const IteratorType & cellU,
                                                     const IteratorType & cellP,
                                                     const unsigned int & f,
                                                     ScratchData<dim> &   scratch_data,
                                                     CopyData &           copy_data_pair) const
{
  boundary_or_uniface_worker<false>(
    cellU, cellP, f, numbers::invalid_unsigned_int, scratch_data, copy_data_pair);

  // auto & [copy_data, copy_data_flipped] = copy_data_pair;

  // /// Velocity "U" takes test function role (in flipped mode ansatz function)
  // auto & phiU = scratch_data.fe_interface_values_test;
  // phiU.reinit(cellU, f);
  // const auto n_dofsU = phiU.n_current_interface_dofs();

  // /// Pressure "P" takes ansatz function role (in flipped mode test function)
  // auto & phiP = scratch_data.fe_interface_values_ansatz;
  // phiP.reinit(cellP, f);
  // const auto n_dofsP = phiP.n_current_interface_dofs();

  // auto & face_data         = copy_data.face_data.emplace_back(n_dofsU, n_dofsP);
  // auto & face_data_flipped = copy_data_flipped.face_data.emplace_back(n_dofsP, n_dofsU);

  // face_data.dof_indices        = phiU.get_interface_dof_indices();
  // face_data.dof_indices_column = phiP.get_interface_dof_indices();

  // AssertDimension(phiU.n_quadrature_points, phiP.n_quadrature_points);
  // std::vector<double> velocity_solution_dot_normals;
  // if(!is_multigrid)
  // {
  //   Assert(analytical_solutionU, ExcMessage("analytical_solutionU is not set."));
  //   AssertDimension(analytical_solutionU->n_components, dim);
  //   const auto &                        q_points = phiU.get_quadrature_points();
  //   const std::vector<Tensor<1, dim>> & normals  = phiU.get_normal_vectors();
  //   std::transform(q_points.cbegin(),
  //                  q_points.cend(),
  //                  normals.cbegin(),
  //                  std::back_inserter(velocity_solution_dot_normals),
  //                  [this](const auto & x_q, const auto & normal) {
  //                    Tensor<1, dim> u_q;
  //                    for(auto c = 0U; c < dim; ++c)
  //                      u_q[c] = analytical_solutionU->value(x_q, c);
  //                    return u_q * normal;
  //                  });
  // }

  // double integral_ijq = 0.;
  // double integral_jq  = 0.;
  // for(unsigned int q = 0; q < phiP.n_quadrature_points; ++q)
  // {
  //   for(unsigned int j = 0; j < n_dofsP; ++j)
  //   {
  //     const auto & av_phiP_j = phiP.average(j, q);

  //     /// Nitsche method (weak Dirichlet conditions)
  //     if(!is_multigrid) // here P is test function
  //     {
  //       const auto & u_dot_n =
  //         velocity_solution_dot_normals[q]; // !!! should be zero for hdiv conf method

  //       integral_jq = u_dot_n * av_phiP_j * phiP.JxW(q);

  //       face_data_flipped.rhs(j) += integral_jq;
  //     }

  //     for(unsigned int i = 0; i < n_dofsU; ++i)
  //     {
  //       /// IP method
  //       const auto & jump_phiU_i_dot_n = compute_vjump_dot_normal(phiU, i, q);

  //       integral_ijq = av_phiP_j * jump_phiU_i_dot_n * phiU.JxW(q);

  //       face_data.matrix(i, j) += integral_ijq;
  //     }
  //   }
  // }

  // /// The pressure-velocity block ("flipped") is the transpose of the
  // /// velocity-pressure block.
  // for(unsigned int i = 0; i < n_dofsU; ++i)
  //   for(unsigned int j = 0; j < n_dofsP; ++j)
  //     face_data_flipped.matrix(j, i) = face_data.matrix(i, j);
}

} // namespace Mixed

} // namespace MW



namespace FD
{
/**
 * Assembles the (exact) local matrices/solvers by exploiting the tensor
 * structure of each scalar-valued shape function, that is each block of a
 * patch matrix involving a component of the velocity vector-field and/or a
 * pressure function have a low-rank Kronecker product decomposition (representation?).
 *
 * However, each block matrix itself has no low-rank Kronecker decomposition
 * (representation?), thus, local matrices are stored and inverted in a
 * standard (vectorized) fashion.
 */
template<int dim,
         int fe_degree_p,
         typename Number              = double,
         TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::Q,
         int             fe_degree_v  = fe_degree_p + 1>
class MatrixIntegratorTensor
{
public:
  using This = MatrixIntegratorTensor<dim, fe_degree_p, Number>;

  static constexpr int n_q_points_1d =
    fe_degree_v + 1 + (dof_layout_v == TPSS::DoFLayout::RT ? 1 : 0);

  using value_type              = Number;
  using transfer_type           = typename TPSS::PatchTransferBlock<dim, Number>;
  using matrix_type_1d          = Table<2, VectorizedArray<Number>>;
  using matrix_type             = MatrixAsTable<VectorizedArray<Number>>;
  using matrix_type_mixed       = Tensors::BlockMatrix<dim, VectorizedArray<Number>, -1, -1>;
  using velocity_evaluator_type = FDEvaluation<dim, fe_degree_v, n_q_points_1d, Number>;
  using pressure_evaluator_type = FDEvaluation<dim, fe_degree_p, n_q_points_1d, Number>;

  void
  initialize(const EquationData & equation_data_in)
  {
    equation_data = equation_data_in;
  }

  template<typename OperatorType>
  void
  assemble_subspace_inverses(const SubdomainHandler<dim, Number> &       subdomain_handler,
                             std::vector<matrix_type> &                  local_matrices,
                             const OperatorType &                        dummy_operator,
                             const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    AssertDimension(subdomain_handler.get_partition_data().n_subdomains(), local_matrices.size());

    using MatrixIntegratorVelocity =
      Velocity::SIPG::FD::MatrixIntegrator<dim, fe_degree_v, Number, dof_layout_v>;
    static_assert(std::is_same<typename MatrixIntegratorVelocity::evaluator_type,
                               velocity_evaluator_type>::value,
                  "Velocity evaluator types mismatch.");
    using matrix_type_velocity = typename MatrixIntegratorVelocity::matrix_type;

    /// Assemble local matrices for the local velocity-velocity block.
    std::vector<matrix_type_velocity> local_matrices_velocity(local_matrices.size());
    {
      MatrixIntegratorVelocity matrix_integrator;
      matrix_integrator.initialize(equation_data);

      matrix_integrator.template assemble_subspace_inverses<OperatorType>(subdomain_handler,
                                                                          local_matrices_velocity,
                                                                          dummy_operator,
                                                                          subdomain_range);
    }

    /// Assemble local matrices for the local pressure-pressure block
    {
      /// This block is zero.
    }

    /// Assemble local matrices for the local velocity-pressure block
    std::vector<matrix_type_mixed> local_matrices_velocity_pressure(local_matrices.size());
    {
      assemble_mixed_subspace_inverses<OperatorType>(subdomain_handler,
                                                     local_matrices_velocity_pressure,
                                                     dummy_operator,
                                                     subdomain_range);
    }

    AssertDimension(local_matrices_velocity.size(), local_matrices.size());
    const auto patch_transfer = get_patch_transfer(subdomain_handler);
    for(auto patch_index = subdomain_range.first; patch_index < subdomain_range.second;
        ++patch_index)
    {
      const auto & local_block_velocity          = local_matrices_velocity[patch_index];
      const auto & local_block_velocity_pressure = local_matrices_velocity_pressure[patch_index];

      patch_transfer->reinit(patch_index);
      const auto n_dofs          = patch_transfer->n_dofs_per_patch();
      const auto n_dofs_velocity = local_block_velocity.m();
      const auto n_dofs_pressure = local_block_velocity_pressure.n();
      AssertDimension(patch_transfer->n_dofs_per_patch(0), n_dofs_velocity);
      (void)n_dofs_pressure;
      AssertDimension(patch_transfer->n_dofs_per_patch(1), n_dofs_pressure);

      auto & local_matrix = local_matrices[patch_index];
      local_matrix.as_table().reinit(n_dofs, n_dofs);

      /// velocity-velocity
      local_matrix.fill_submatrix(local_block_velocity.as_table(), 0U, 0U);

      /// velocity-pressure
      local_matrix.fill_submatrix(local_block_velocity_pressure.as_table(), 0U, n_dofs_velocity);

      /// pressure-velocity
      local_matrix.template fill_submatrix<true>(local_block_velocity_pressure.as_table(),
                                                 n_dofs_velocity,
                                                 0U);

      local_matrix.invert({equation_data.local_kernel_size, equation_data.local_kernel_threshold});
    }
  }

  template<typename OperatorType>
  void
  assemble_mixed_subspace_inverses(
    const SubdomainHandler<dim, Number> & subdomain_handler,
    std::vector<matrix_type_mixed> &      local_matrices,
    const OperatorType &,
    const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    AssertDimension(subdomain_handler.get_partition_data().n_subdomains(), local_matrices.size());

    for(auto compU = 0U; compU < dim; ++compU)
    {
      velocity_evaluator_type eval_velocity(subdomain_handler, /*dofh_index*/ 0, compU);
      pressure_evaluator_type eval_pressure(subdomain_handler, /*dofh_index*/ 1, /*component*/ 0);

      for(unsigned int patch = subdomain_range.first; patch < subdomain_range.second; ++patch)
      {
        auto & velocity_pressure_matrix = local_matrices[patch];
        if(velocity_pressure_matrix.n_block_rows() == 0 &&
           velocity_pressure_matrix.n_block_cols() == 0)
          velocity_pressure_matrix.resize(dim, 1);

        eval_velocity.reinit(patch);
        eval_pressure.reinit(patch);

        const auto mass_matrices =
          assemble_mass_tensor(/*test*/ eval_velocity, /*ansatz*/ eval_pressure);
        /// Note that we have flipped ansatz and test functions roles. The
        /// divergence of the velocity test functions is obtained by
        /// transposing gradient matrices.
        const auto gradient_matrices =
          assemble_gradient_tensor(/*test*/ eval_pressure, /*ansatz*/ eval_velocity);

        const auto & MxMxGT = [&](const unsigned int direction_of_div) {
          /// For example, we obtain MxMxGT for direction_of_div = 0 (dimension
          /// 0 is rightmost!)
          std::array<matrix_type_1d, dim> kronecker_tensor;
          for(auto d = 0U; d < dim; ++d)
            kronecker_tensor[d] = d == direction_of_div ?
                                    -1. * LinAlg::transpose(gradient_matrices[direction_of_div]) :
                                    mass_matrices[d];
          return kronecker_tensor;
        };

        std::vector<std::array<matrix_type_1d, dim>> rank1_tensors;
        rank1_tensors.emplace_back(MxMxGT(compU));
        velocity_pressure_matrix.get_block(compU, 0).reinit(rank1_tensors);
      }
    }
  }

  std::array<matrix_type_1d, dim>
  assemble_mass_tensor(velocity_evaluator_type & eval_test,
                       pressure_evaluator_type & eval_ansatz) const
  {
    using CellMass = ::FD::L2::CellOperation<dim, fe_degree_v, n_q_points_1d, Number, fe_degree_p>;
    return eval_test.patch_action(eval_ansatz, CellMass{});
  }

  /**
   * We remark that the velocity takes the ansatz function role here
   * (Gradient::CellOperation derives the ansatz function) although in
   * assemble_mixed_subspace_inverses() we require the divergence of the
   * velocity test functions. Therefore, we transpose the returned tensor of
   * matrices.
   */
  std::array<matrix_type_1d, dim>
  assemble_gradient_tensor(pressure_evaluator_type & eval_test,
                           velocity_evaluator_type & eval_ansatz) const
  {
    using CellGradient =
      ::FD::Gradient::CellOperation<dim, fe_degree_p, n_q_points_1d, Number, fe_degree_v>;
    CellGradient cell_gradient;
    return eval_test.patch_action(eval_ansatz, cell_gradient);
  }

  std::shared_ptr<transfer_type>
  get_patch_transfer(const SubdomainHandler<dim, Number> & subdomain_handler) const
  {
    return std::make_shared<transfer_type>(subdomain_handler);
  }

  EquationData equation_data;
};



/**
 * This class is actual not an "integration-type" struct for local matrices. It
 * simply uses PatchTransferBlock w.r.t. the velocity-pressure block system to
 * extract the local matrices/solvers from the (global) level matrix. The level
 * matrix is passed as argument to assemble_subspace_inverses().
 *
 * Therefore, all local matrices are stored and inverted in a standard way.
 */
template<int dim,
         int fe_degree_p,
         typename Number              = double,
         TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::Q,
         int             fe_degree_v  = fe_degree_p + 1>
class MatrixIntegratorCut
{
public:
  using This = MatrixIntegratorCut<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v>;

  static constexpr int n_q_points_1d =
    fe_degree_v + 1 + (dof_layout_v == TPSS::DoFLayout::RT ? 1 : 0);

  using value_type    = Number;
  using transfer_type = typename TPSS::PatchTransferBlock<dim, Number>;
  using matrix_type   = Tensors::BlockMatrixBasic2x2<MatrixAsTable<VectorizedArray<Number>>>;
  using operator_type = TrilinosWrappers::BlockSparseMatrix;

  void
  initialize(const EquationData & equation_data_in)
  {
    equation_data = equation_data_in;
  }

  void
  assemble_subspace_inverses(const SubdomainHandler<dim, Number> &       subdomain_handler,
                             std::vector<matrix_type> &                  local_matrices,
                             const operator_type &                       level_matrix,
                             const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    AssertDimension(subdomain_handler.get_partition_data().n_subdomains(), local_matrices.size());

    const unsigned int n_mpi_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

    typename matrix_type::AdditionalData additional_data;
    additional_data.basic_inverse = {equation_data.local_kernel_size,
                                     equation_data.local_kernel_threshold};

    const auto   patch_transfer        = get_patch_transfer(subdomain_handler);
    const auto & patch_worker_velocity = patch_transfer->get_patch_dof_worker(0);

    FullMatrix<double> tmp_v_v;
    FullMatrix<double> tmp_p_p;
    FullMatrix<double> tmp_v_p;
    FullMatrix<double> tmp_p_v;

    FullMatrix<double> locally_relevant_level_matrix_v_v;
    FullMatrix<double> locally_relevant_level_matrix_p_p;
    FullMatrix<double> locally_relevant_level_matrix_v_p;
    FullMatrix<double> locally_relevant_level_matrix_p_v;
    if(n_mpi_procs > 1)
    {
      const auto partitioner_v = subdomain_handler.get_vector_partitioner(0);
      const auto partitioner_p = subdomain_handler.get_vector_partitioner(1);

      locally_relevant_level_matrix_v_v =
        std::move(Util::extract_locally_relevant_matrix(level_matrix.block(0, 0), partitioner_v));
      locally_relevant_level_matrix_p_p =
        std::move(Util::extract_locally_relevant_matrix(level_matrix.block(1, 1), partitioner_p));
      locally_relevant_level_matrix_v_p = std::move(Util::extract_locally_relevant_matrix(
        level_matrix.block(0, 1), partitioner_v, partitioner_p));
      locally_relevant_level_matrix_p_v = std::move(Util::extract_locally_relevant_matrix(
        level_matrix.block(1, 0), partitioner_p, partitioner_v));

      /// DEBUG
      // std::ofstream      ofs;
      // std::ostringstream oss;
      // oss << "debug.p" << mpi_rank << ".txt";
      // ofs.open(oss.str(), std::ios_base::out);
      // locally_relevant_level_matrix_v_v.print_formatted(ofs);
    }

    const auto make_local_indices_impl =
      [&](const std::vector<types::global_dof_index> &               indices,
          const std::shared_ptr<const Utilities::MPI::Partitioner> & vector_partitioner) {
        std::vector<unsigned int> local_dof_indices;
        std::transform(indices.begin(),
                       indices.end(),
                       std::back_inserter(local_dof_indices),
                       [&](const auto dof_index) {
                         return vector_partitioner->global_to_local(dof_index);
                       });
        return local_dof_indices;
      };

    for(auto patch_index = subdomain_range.first; patch_index < subdomain_range.second;
        ++patch_index)
    {
      patch_transfer->reinit(patch_index);
      const auto n_dofs          = patch_transfer->n_dofs_per_patch();
      const auto n_dofs_velocity = patch_transfer->n_dofs_per_patch(0);
      const auto n_dofs_pressure = patch_transfer->n_dofs_per_patch(1);

      matrix_type & patch_matrix = local_matrices[patch_index];

      auto & local_block_velocity = patch_matrix.get_block(0U, 0U);
      local_block_velocity.as_table().reinit(n_dofs_velocity, n_dofs_velocity);
      tmp_v_v.reinit(n_dofs_velocity, n_dofs_velocity);

      auto & local_block_pressure = patch_matrix.get_block(1U, 1U);
      local_block_pressure.as_table().reinit(n_dofs_pressure, n_dofs_pressure);
      tmp_p_p.reinit(n_dofs_pressure, n_dofs_pressure);

      auto & local_block_velocity_pressure = patch_matrix.get_block(0U, 1U);
      local_block_velocity_pressure.as_table().reinit(n_dofs_velocity, n_dofs_pressure);
      tmp_v_p.reinit(n_dofs_velocity, n_dofs_pressure);

      auto & local_block_pressure_velocity = patch_matrix.get_block(1U, 0U);
      local_block_pressure_velocity.as_table().reinit(n_dofs_pressure, n_dofs_velocity);
      tmp_p_v.reinit(n_dofs_pressure, n_dofs_velocity);

      for(auto lane = 0U; lane < patch_worker_velocity.n_lanes_filled(patch_index); ++lane)
      {
        /// Patch-wise local and global dof indices of velocity block.
        const auto & patch_transfer_velocity = patch_transfer->get_patch_transfer(0);
        const std::vector<types::global_dof_index> velocity_dof_indices_on_patch =
          std::move(patch_transfer_velocity.get_global_dof_indices(lane));

        /// Patch-wise local and global dof indices of pressure block.
        const auto & patch_transfer_pressure = patch_transfer->get_patch_transfer(1);
        const std::vector<types::global_dof_index> pressure_dof_indices_on_patch =
          std::move(patch_transfer_pressure.get_global_dof_indices(lane));

        const auto local_dof_indices_v =
          std::move(make_local_indices_impl(velocity_dof_indices_on_patch,
                                            subdomain_handler.get_vector_partitioner(0)));

        const auto local_dof_indices_p =
          std::move(make_local_indices_impl(pressure_dof_indices_on_patch,
                                            subdomain_handler.get_vector_partitioner(1)));

        /// velocity block
        if(n_mpi_procs > 1) /// parallel
        {
          AssertThrow(equation_data.local_solver == LocalSolver::Exact, ExcMessage("TODO..."));
          tmp_v_v.extract_submatrix_from(locally_relevant_level_matrix_v_v,
                                         local_dof_indices_v,
                                         local_dof_indices_v);
          local_block_velocity.fill_submatrix(tmp_v_v, 0U, 0U, lane);
        }

        else /// serial
        {
          if(equation_data.local_solver == LocalSolver::Vdiag)
          {
            for(auto comp = 0U; comp < dim; ++comp)
            {
              std::vector<types::global_dof_index> velocity_dof_indices_per_comp;
              const auto view = patch_transfer_velocity.get_dof_indices(lane, comp);
              std::copy(view.cbegin(),
                        view.cend(),
                        std::back_inserter(velocity_dof_indices_per_comp));
              const auto n_velocity_dofs_per_comp = velocity_dof_indices_per_comp.size();

              tmp_v_v.reinit(n_velocity_dofs_per_comp, n_velocity_dofs_per_comp);
              tmp_v_v.extract_submatrix_from(level_matrix.block(0U, 0U),
                                             velocity_dof_indices_per_comp,
                                             velocity_dof_indices_per_comp);

              const auto start = comp * n_velocity_dofs_per_comp;
              local_block_velocity.fill_submatrix(tmp_v_v, start, start, lane);
            }
          }
          else
          {
            tmp_v_v.extract_submatrix_from(level_matrix.block(0U, 0U),
                                           velocity_dof_indices_on_patch,
                                           velocity_dof_indices_on_patch);
            local_block_velocity.fill_submatrix(tmp_v_v, 0U, 0U, lane);
          }
        }

        /// pressure block
        if(n_mpi_procs > 1) /// parallel
          tmp_p_p.extract_submatrix_from(locally_relevant_level_matrix_p_p,
                                         local_dof_indices_p,
                                         local_dof_indices_p);
        else /// serial
          tmp_p_p.extract_submatrix_from(level_matrix.block(1U, 1U),
                                         pressure_dof_indices_on_patch,
                                         pressure_dof_indices_on_patch);
        local_block_pressure.fill_submatrix(tmp_p_p, 0U, 0U, lane);

        /// velocity-pressure block
        if(n_mpi_procs > 1) /// parallel
          tmp_v_p.extract_submatrix_from(locally_relevant_level_matrix_v_p,
                                         local_dof_indices_v,
                                         local_dof_indices_p);
        else
          tmp_v_p.extract_submatrix_from(level_matrix.block(0U, 1U),
                                         velocity_dof_indices_on_patch,
                                         pressure_dof_indices_on_patch);
        local_block_velocity_pressure.fill_submatrix(tmp_v_p, 0U, 0U, lane);

        /// pressure-velocity block
        if(n_mpi_procs > 1) /// parallel
          tmp_p_v.extract_submatrix_from(locally_relevant_level_matrix_p_v,
                                         local_dof_indices_p,
                                         local_dof_indices_v);
        else
          tmp_p_v.extract_submatrix_from(level_matrix.block(1U, 0U),
                                         pressure_dof_indices_on_patch,
                                         velocity_dof_indices_on_patch);
        local_block_pressure_velocity.fill_submatrix(tmp_p_v, 0U, 0U, lane);
      }

      (void)n_dofs;
      AssertDimension(patch_matrix.m(), n_dofs);
      AssertDimension(patch_matrix.n(), n_dofs);

      patch_matrix.invert(additional_data);
    }
  }

  std::shared_ptr<transfer_type>
  get_patch_transfer(const SubdomainHandler<dim, Number> & subdomain_handler) const
  {
    return std::make_shared<transfer_type>(subdomain_handler);
  }

  EquationData       equation_data;
  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
};



/**
 * This class actually makes no use of fast diagonalization but simply uses the
 * MeshWorker framework to assemble local matrices. Nevertheless
 * PatchTransferBlock is used as transfer and its underlying PatchDoFWorker to
 * determine collections of cell iterators.
 *
 * Therefore, all local matrices are stored and inverted in a standard way, that
 * is without exploiting any tensor structure.
 */
template<int dim,
         int fe_degree_p,
         typename Number              = double,
         TPSS::DoFLayout dof_layout_v = TPSS::DoFLayout::Q,
         int             fe_degree_v  = fe_degree_p + 1>
class MatrixIntegratorLMW
{
public:
  using This = MatrixIntegratorLMW<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v>;

  static constexpr int n_q_points_1d =
    fe_degree_v + 1 + (dof_layout_v == TPSS::DoFLayout::RT ? 1 : 0);
  static constexpr bool use_sipg_method     = dof_layout_v == TPSS::DoFLayout::DGQ;
  static constexpr bool use_hdivsipg_method = dof_layout_v == TPSS::DoFLayout::RT;
  static constexpr bool use_conf_method     = dof_layout_v == TPSS::DoFLayout::Q;

  using value_type    = Number;
  using transfer_type = typename TPSS::PatchTransferBlock<dim, Number>;
  using matrix_type   = Tensors::BlockMatrixBasic2x2<MatrixAsTable<VectorizedArray<Number>>>;
  using operator_type = TrilinosWrappers::BlockSparseMatrix;

  void
  initialize(const EquationData & equation_data_in)
  {
    equation_data = equation_data_in;
  }

  void
  assemble_subspace_inverses(const SubdomainHandler<dim, Number> & subdomain_handler,
                             std::vector<matrix_type> &            local_matrices,
                             const operator_type & /*level_matrix*/,
                             const std::pair<unsigned int, unsigned int> subdomain_range) const
  {
    AssertDimension(subdomain_handler.get_partition_data().n_subdomains(), local_matrices.size());

    typename matrix_type::AdditionalData additional_data;
    additional_data.basic_inverse = {equation_data.local_kernel_size,
                                     equation_data.local_kernel_threshold};

    const auto   patch_transfer     = get_patch_transfer(subdomain_handler);
    const auto & patch_dof_worker_v = patch_transfer->get_patch_dof_worker(0);
    const auto & patch_dof_worker_p = patch_transfer->get_patch_dof_worker(1);

    FullMatrix<double> tmp_v_v;
    FullMatrix<double> tmp_p_p;
    FullMatrix<double> tmp_v_p;
    FullMatrix<double> tmp_p_v;

    const auto [begin, end] = subdomain_range;
    for(auto patch_index = begin; patch_index < end; ++patch_index)
    {
      patch_transfer->reinit(patch_index);

      const auto n_dofs          = patch_transfer->n_dofs_per_patch();
      const auto n_dofs_velocity = patch_transfer->n_dofs_per_patch(0);
      const auto n_dofs_pressure = patch_transfer->n_dofs_per_patch(1);

      const auto & patch_transfer_v = patch_transfer->get_patch_transfer(0);
      const auto & patch_transfer_p = patch_transfer->get_patch_transfer(1);

      matrix_type & patch_matrix = local_matrices[patch_index];

      auto & block_velocity = patch_matrix.get_block(0U, 0U);
      block_velocity.as_table().reinit(n_dofs_velocity, n_dofs_velocity);
      tmp_v_v.reinit(n_dofs_velocity, n_dofs_velocity);

      auto & block_pressure = patch_matrix.get_block(1U, 1U);
      block_pressure.as_table().reinit(n_dofs_pressure, n_dofs_pressure);
      tmp_p_p.reinit(n_dofs_pressure, n_dofs_pressure);

      auto & block_velocity_pressure = patch_matrix.get_block(0U, 1U);
      block_velocity_pressure.as_table().reinit(n_dofs_velocity, n_dofs_pressure);
      tmp_v_p.reinit(n_dofs_velocity, n_dofs_pressure);

      auto & block_pressure_velocity = patch_matrix.get_block(1U, 0U);
      block_pressure_velocity.as_table().reinit(n_dofs_pressure, n_dofs_velocity);
      tmp_p_v.reinit(n_dofs_pressure, n_dofs_velocity);

      for(auto lane = 0U; lane < patch_dof_worker_v.n_lanes_filled(patch_index); ++lane)
      {
        /// velocity block
        {
          using Velocity::SIPG::MW::CopyData;
          using Velocity::SIPG::MW::ScratchData;
          using MatrixIntegrator   = Velocity::SIPG::MW::MatrixIntegrator<dim, true>;
          using cell_iterator_type = typename MatrixIntegrator::IteratorType;

          tmp_v_v = 0.;

          const auto & cell_collection = patch_dof_worker_v.get_cell_collection(patch_index, lane);

          const TPSS::BelongsToCollection<cell_iterator_type> belongs_to_collection(
            cell_collection);

          const auto & local_cell_range = TPSS::make_local_cell_range(cell_collection);

          const auto & g2l = patch_transfer_v.get_global_to_local_dof_indices(lane);

          const auto distribute_local_to_patch_impl = [&](const auto & cd) {
            std::vector<unsigned int> local_dof_indices;
            std::transform(cd.dof_indices.begin(),
                           cd.dof_indices.end(),
                           std::back_inserter(local_dof_indices),
                           [&](const auto dof_index) {
                             const auto & local_index = g2l.find(dof_index);
                             return local_index != g2l.cend() ? local_index->second :
                                                                numbers::invalid_unsigned_int;
                           });
            for(auto i = 0U; i < cd.matrix.m(); ++i)
              if(local_dof_indices[i] != numbers::invalid_unsigned_int)
                for(auto j = 0U; j < cd.matrix.n(); ++j)
                  if(local_dof_indices[j] != numbers::invalid_unsigned_int)
                    tmp_v_v(local_dof_indices[i], local_dof_indices[j]) += cd.matrix(i, j);
          };

          const auto local_copier = [&](const CopyData & copy_data) {
            for(const auto & cd : copy_data.cell_data)
              distribute_local_to_patch_impl(cd);
            for(const auto & cdf : copy_data.face_data)
              distribute_local_to_patch_impl(cdf);
          };

          const MatrixIntegrator matrix_integrator(nullptr, nullptr, nullptr, equation_data);

          const UpdateFlags update_flags =
            update_values | update_gradients | update_quadrature_points | update_JxW_values;
          const UpdateFlags interface_update_flags = update_flags | update_normal_vectors;
          ScratchData<dim>  scratch_data(subdomain_handler.get_mapping(),
                                        subdomain_handler.get_dof_handler(0).get_fe(),
                                        subdomain_handler.get_dof_handler(0).get_fe(),
                                        n_q_points_1d,
                                        update_flags,
                                        update_flags,
                                        interface_update_flags,
                                        interface_update_flags);

          CopyData copy_data;

          if(use_conf_method)
            MeshWorker::m2d2::mesh_loop(
              local_cell_range,
              [&](const auto & cell, auto & scratch_data, auto & copy_data) {
                matrix_integrator.cell_worker(cell, scratch_data, copy_data);
              },
              local_copier,
              scratch_data,
              copy_data,
              MeshWorker::assemble_own_cells | MeshWorker::assemble_ghost_cells);

          else if(use_sipg_method || use_hdivsipg_method)
            MeshWorker::m2d2::mesh_loop(
              local_cell_range,
              [&](const auto & cell, auto & scratch_data, auto & copy_data) {
                matrix_integrator.cell_worker(cell, scratch_data, copy_data);
              },
              local_copier,
              scratch_data,
              copy_data,
              MeshWorker::assemble_own_cells | MeshWorker::assemble_ghost_cells |
                MeshWorker::assemble_boundary_faces | MeshWorker::assemble_own_interior_faces_both |
                MeshWorker::assemble_ghost_faces_both,
              /*assemble faces at ghosts?*/ true,
              [&](const auto & cell, const auto face_no, auto & scratch_data, auto & copy_data) {
                if(use_sipg_method)
                  matrix_integrator.boundary_worker(cell, face_no, scratch_data, copy_data);
                else if(use_hdivsipg_method)
                  matrix_integrator.boundary_worker_tangential(cell,
                                                               face_no,
                                                               scratch_data,
                                                               copy_data);
              },
              [&](const auto & cell,
                  const auto   face_no,
                  const auto   sface_no,
                  const auto & ncell,
                  const auto   nface_no,
                  const auto   nsface_no,
                  auto &       scratch_data,
                  auto &       copy_data) {
                const bool cell_belongs_to_collection  = belongs_to_collection(cell);
                const bool ncell_belongs_to_collection = belongs_to_collection(ncell);
                const bool is_interface = cell_belongs_to_collection && ncell_belongs_to_collection;
                if(is_interface)
                {
                  if(use_sipg_method)
                    matrix_integrator.face_worker(
                      cell, face_no, sface_no, ncell, nface_no, nsface_no, scratch_data, copy_data);
                  else if(use_hdivsipg_method)
                    matrix_integrator.face_worker_tangential(
                      cell, face_no, sface_no, ncell, nface_no, nsface_no, scratch_data, copy_data);
                  copy_data.face_data.back().matrix *= 0.5; /// both sides!
                  return;
                }
                if(cell_belongs_to_collection)
                {
                  if(use_sipg_method)
                    matrix_integrator.uniface_worker(
                      cell, face_no, sface_no, scratch_data, copy_data);
                  else if(use_hdivsipg_method)
                    matrix_integrator.uniface_worker_tangential(
                      cell, face_no, sface_no, scratch_data, copy_data);
                }
              });

          else
            Assert(false, ExcMessage("FEM is not implemented."));
        }

        if(equation_data.local_solver == LocalSolver::Vdiag)
        {
          for(auto comp = 0U; comp < dim; ++comp)
          {
            const std::vector<types::global_dof_index> & velocity_dof_indices_per_comp =
              patch_transfer_v.get_global_dof_indices(lane, comp);
            const auto n_velocity_dofs_per_comp = velocity_dof_indices_per_comp.size();

            const unsigned int start = comp * n_velocity_dofs_per_comp;

            std::vector<unsigned int> local_dof_indices(n_velocity_dofs_per_comp);
            std::iota(local_dof_indices.begin(), local_dof_indices.end(), start);

            FullMatrix<double> tmp_per_comp(n_velocity_dofs_per_comp);
            tmp_per_comp.extract_submatrix_from(tmp_v_v, local_dof_indices, local_dof_indices);

            block_velocity.fill_submatrix(tmp_per_comp, start, start, lane);
          }
        }
        else if(equation_data.local_solver == LocalSolver::Exact)
          block_velocity.fill_submatrix(tmp_v_v, 0U, 0U, lane);
        else
          Assert(false, ExcMessage("local solver variant not implemented"));

        /// pressure block (just fill with zeros!)
        block_pressure.fill_submatrix(tmp_p_p, 0U, 0U, lane);

        /// velocity-pressure block & pressure-velocity block
        {
          using VelocityPressure::MW::Mixed::CopyData;
          using VelocityPressure::MW::Mixed::ScratchData;
          using MatrixIntegrator   = VelocityPressure::MW::Mixed::MatrixIntegrator<dim, true>;
          using cell_iterator_type = typename MatrixIntegrator::IteratorType;

          tmp_v_p = 0.;
          tmp_p_v = 0.;

          const auto & dof_handler_velocity = subdomain_handler.get_dof_handler(0);
          const auto & dof_handler_pressure = subdomain_handler.get_dof_handler(1);

          const auto & cell_collection_v =
            patch_dof_worker_v.get_cell_collection(patch_index, lane);
          const auto & cell_collection_p =
            patch_dof_worker_p.get_cell_collection(patch_index, lane);

          const auto & local_cell_range_v = TPSS::make_local_cell_range(cell_collection_v);

          const TPSS::BelongsToCollection<cell_iterator_type> belongs_to_collection_v(
            cell_collection_v);

          const auto & g2l_v = patch_transfer_v.get_global_to_local_dof_indices(lane);
          const auto & g2l_p = patch_transfer_p.get_global_to_local_dof_indices(lane);

          const auto distribute_local_to_patch_impl = [&](const auto & cd,
                                                          const auto & cd_flipped) {
            std::vector<unsigned int> local_dof_indices_v;
            std::transform(cd.dof_indices.begin(),
                           cd.dof_indices.end(),
                           std::back_inserter(local_dof_indices_v),
                           [&](const auto dof_index) {
                             const auto & local_index = g2l_v.find(dof_index);
                             return local_index != g2l_v.cend() ? local_index->second :
                                                                  numbers::invalid_unsigned_int;
                           });
            std::vector<unsigned int> local_dof_indices_p;
            std::transform(cd.dof_indices_column.begin(),
                           cd.dof_indices_column.end(),
                           std::back_inserter(local_dof_indices_p),
                           [&](const auto dof_index) {
                             const auto & local_index = g2l_p.find(dof_index);
                             return local_index != g2l_p.cend() ? local_index->second :
                                                                  numbers::invalid_unsigned_int;
                           });
            AssertDimension(cd.matrix.m(), local_dof_indices_v.size());
            AssertDimension(cd.matrix.n(), local_dof_indices_p.size());
            /// velocity-pressure
            for(auto i = 0U; i < cd.matrix.m(); ++i)
              if(local_dof_indices_v[i] != numbers::invalid_unsigned_int)
                for(auto j = 0U; j < cd.matrix.n(); ++j)
                  if(local_dof_indices_p[j] != numbers::invalid_unsigned_int)
                    tmp_v_p(local_dof_indices_v[i], local_dof_indices_p[j]) += cd.matrix(i, j);
            AssertDimension(cd_flipped.matrix.m(), local_dof_indices_p.size());
            AssertDimension(cd_flipped.matrix.n(), local_dof_indices_v.size());
            /// pressure-velocity
            for(auto i = 0U; i < cd_flipped.matrix.m(); ++i)
              if(local_dof_indices_p[i] != numbers::invalid_unsigned_int)
                for(auto j = 0U; j < cd_flipped.matrix.n(); ++j)
                  if(local_dof_indices_v[j] != numbers::invalid_unsigned_int)
                    tmp_p_v(local_dof_indices_p[i], local_dof_indices_v[j]) +=
                      cd_flipped.matrix(i, j);
          };

          const auto local_copier = [&](const CopyData & copy_data_pair) {
            const auto & [copy_data, copy_data_flipped] = copy_data_pair;

            AssertDimension(copy_data.cell_data.size(), copy_data_flipped.cell_data.size());
            AssertDimension(copy_data.face_data.size(), copy_data_flipped.face_data.size());

            auto cd_flipped = copy_data_flipped.cell_data.cbegin();
            auto cd         = copy_data.cell_data.cbegin();
            for(; cd != copy_data.cell_data.cend(); ++cd, ++cd_flipped)
              distribute_local_to_patch_impl(*cd, *cd_flipped);

            auto cdf_flipped = copy_data_flipped.face_data.cbegin();
            auto cdf         = copy_data.face_data.cbegin();
            for(; cdf != copy_data.face_data.cend(); ++cdf, ++cdf_flipped)
              distribute_local_to_patch_impl(*cdf, *cdf_flipped);
          };

          const MatrixIntegrator matrix_integrator(
            nullptr, nullptr, nullptr, nullptr, equation_data);

          const UpdateFlags update_flags_velocity =
            update_values | update_gradients | update_quadrature_points | update_JxW_values;
          const UpdateFlags update_flags_pressure =
            update_values | update_gradients | update_quadrature_points | update_JxW_values;
          const UpdateFlags interface_update_flags_velocity =
            update_values | update_quadrature_points | update_JxW_values | update_normal_vectors;
          const UpdateFlags interface_update_flags_pressure =
            update_values | update_quadrature_points | update_JxW_values | update_normal_vectors;

          ScratchData<dim> scratch_data(subdomain_handler.get_mapping(),
                                        dof_handler_velocity.get_fe(),
                                        dof_handler_pressure.get_fe(),
                                        n_q_points_1d,
                                        update_flags_velocity,
                                        update_flags_pressure,
                                        interface_update_flags_velocity,
                                        interface_update_flags_pressure);

          CopyData copy_data;

          const auto & cell_worker =
            [&](const cell_iterator_type & cell, auto & scratch_data, auto & copy_data) {
              cell_iterator_type cell_ansatz(&(dof_handler_pressure.get_triangulation()),
                                             cell->level(),
                                             cell->index(),
                                             &dof_handler_pressure);
              matrix_integrator.cell_worker(cell, cell_ansatz, scratch_data, copy_data);
            };

          if(use_conf_method || use_hdivsipg_method)
            MeshWorker::m2d2::mesh_loop(local_cell_range_v,
                                        cell_worker,
                                        local_copier,
                                        scratch_data,
                                        copy_data,
                                        MeshWorker::assemble_own_cells |
                                          MeshWorker::assemble_ghost_cells);

          else if(use_sipg_method)
            MeshWorker::m2d2::mesh_loop(
              local_cell_range_v,
              cell_worker,
              local_copier,
              scratch_data,
              copy_data,
              MeshWorker::assemble_own_cells | MeshWorker::assemble_ghost_cells |
                MeshWorker::assemble_boundary_faces | MeshWorker::assemble_own_interior_faces_both |
                MeshWorker::assemble_ghost_faces_both,
              /*assemble faces at ghosts?*/ true,
              [&](const auto & cell, const auto face_no, auto & scratch_data, auto & copy_data) {
                cell_iterator_type cell_ansatz(&(dof_handler_pressure.get_triangulation()),
                                               cell->level(),
                                               cell->index(),
                                               &dof_handler_pressure);
                matrix_integrator.boundary_worker(
                  cell, cell_ansatz, face_no, scratch_data, copy_data);
              },
              [&](const auto & cell,
                  const auto   face_no,
                  const auto   sface_no,
                  const auto & ncell,
                  const auto   nface_no,
                  const auto   nsface_no,
                  auto &       scratch_data,
                  auto &       copy_data_pair) {
                cell_iterator_type cell_ansatz(&(dof_handler_pressure.get_triangulation()),
                                               cell->level(),
                                               cell->index(),
                                               &dof_handler_pressure);
                cell_iterator_type ncell_ansatz(&(dof_handler_pressure.get_triangulation()),
                                                ncell->level(),
                                                ncell->index(),
                                                &dof_handler_pressure);
                const bool         cell_belongs_to_collection  = belongs_to_collection_v(cell);
                const bool         ncell_belongs_to_collection = belongs_to_collection_v(ncell);
                const bool is_interface = cell_belongs_to_collection && ncell_belongs_to_collection;
                if(is_interface)
                {
                  matrix_integrator.face_worker(cell,
                                                cell_ansatz,
                                                face_no,
                                                sface_no,
                                                ncell,
                                                ncell_ansatz,
                                                nface_no,
                                                nsface_no,
                                                scratch_data,
                                                copy_data_pair);
                  /// interfaces are assembled from both sides
                  auto & [copy_data, copy_data_flipped] = copy_data_pair;
                  copy_data.face_data.back().matrix *= 0.5;
                  copy_data_flipped.face_data.back().matrix *= 0.5;
                  return;
                }
                if(cell_belongs_to_collection)
                {
                  matrix_integrator.uniface_worker(
                    cell, cell_ansatz, face_no, sface_no, scratch_data, copy_data_pair);
                }
              });

          else
            Assert(false, ExcMessage("FEM is not implemented."));

          block_velocity_pressure.fill_submatrix(tmp_v_p, 0U, 0U, lane);

          block_pressure_velocity.fill_submatrix(tmp_p_v, 0U, 0U, lane);
        }
      }

      (void)n_dofs;
      AssertDimension(patch_matrix.m(), n_dofs);
      AssertDimension(patch_matrix.n(), n_dofs);

      patch_matrix.invert(additional_data);
    }
  }

  std::shared_ptr<transfer_type>
  get_patch_transfer(const SubdomainHandler<dim, Number> & subdomain_handler) const
  {
    return std::make_shared<transfer_type>(subdomain_handler);
  }

  EquationData equation_data;
};



/**
 * Selects the MatrixIntegrator at compile time w.r.t. the LocalAssembly
 * template. Hence, the generic class is empty and we select by template
 * specialization.
 */
template<LocalAssembly local_assembly,
         int           dim,
         int           fe_degree_p,
         typename Number,
         TPSS::DoFLayout dof_layout_v,
         int             fe_degree_v>
struct MatrixIntegratorSelector
{
};

template<int dim, int fe_degree_p, typename Number, TPSS::DoFLayout dof_layout_v, int fe_degree_v>
struct MatrixIntegratorSelector<LocalAssembly::Tensor,
                                dim,
                                fe_degree_p,
                                Number,
                                dof_layout_v,
                                fe_degree_v>
{
  using type = MatrixIntegratorTensor<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v>;
};

template<int dim, int fe_degree_p, typename Number, TPSS::DoFLayout dof_layout_v, int fe_degree_v>
struct MatrixIntegratorSelector<LocalAssembly::Cut,
                                dim,
                                fe_degree_p,
                                Number,
                                dof_layout_v,
                                fe_degree_v>
{
  using type = MatrixIntegratorCut<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v>;
};

template<int dim, int fe_degree_p, typename Number, TPSS::DoFLayout dof_layout_v, int fe_degree_v>
struct MatrixIntegratorSelector<LocalAssembly::LMW,
                                dim,
                                fe_degree_p,
                                Number,
                                dof_layout_v,
                                fe_degree_v>
{
  using type = MatrixIntegratorLMW<dim, fe_degree_p, Number, dof_layout_v, fe_degree_v>;
};



/**
 * Aliasing MatrixIntegrator w.r.t. the choice of LocalAssembly.
 */
template<int dim,
         int fe_degree_p,
         typename Number                = double,
         TPSS::DoFLayout dof_layout_v   = TPSS::DoFLayout::Q,
         int             fe_degree_v    = fe_degree_p + 1,
         LocalAssembly   local_assembly = LocalAssembly::Tensor>
using MatrixIntegrator = typename MatrixIntegratorSelector<local_assembly,
                                                           dim,
                                                           fe_degree_p,
                                                           Number,
                                                           dof_layout_v,
                                                           fe_degree_v>::type;

} // end namespace FD

} // namespace VelocityPressure

} // namespace Stokes

#endif /* APPS_STOKESINTEGRATOR_H_ */
