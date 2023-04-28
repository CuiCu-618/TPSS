/**
 * utilites.h
 *
 * collection of helper functions
 *
 *  Created on: Sep 26, 2019
 *      Author: witte
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/hp/fe_values.h>

#include "git_version.h"
#include "solvers_and_preconditioners/TPSS/generic_functionalities.h"
#include "solvers_and_preconditioners/TPSS/tensors.h"


#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>



using namespace dealii;

namespace Util
{
template<typename T>
std::string
parameter_to_fstring(const std::string & description, const T parameter)
{
  AssertIndexRange(description.size(), 43);
  std::ostringstream oss;
  print_row_variable(oss, 2, "", 43, description, parameter);
  return oss.str();
}



std::string
git_version_to_fstring()
{
  std::ostringstream oss;
  oss << parameter_to_fstring("Git - deal.II version: ", DEAL_II_GIT_SHORTREV);
  oss << parameter_to_fstring("Git - deal.II branch: ", DEAL_II_GIT_BRANCH);
  oss << parameter_to_fstring("Git - TPSS version: ", GIT_COMMIT_HASH);
  oss << parameter_to_fstring("Git - TPSS branch: ", GIT_BRANCH);
  return oss.str();
}



std::string
generic_info_to_fstring()
{
  std::ostringstream oss;
  oss << Util::git_version_to_fstring();
  oss << Util::parameter_to_fstring("Date:", Utilities::System::get_date());
  oss << Util::parameter_to_fstring("Number of MPI processes:",
                                    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
  oss << Util::parameter_to_fstring("Number of threads per MPI proc:",
                                    MultithreadInfo::n_threads());
  oss << Util::parameter_to_fstring("Vectorization level:",
                                    Utilities::System::get_current_vectorization_level());
  const auto size_of_global_dof_index = sizeof(types::global_dof_index{0});
  oss << Util::parameter_to_fstring("Size of global_dof_index (bits):",
                                    8 * size_of_global_dof_index);
  return oss.str();
}



static constexpr char const * skipper = "o";

std::vector<char const *>
args_to_strings(const int argc_in, char * argv_in[])
{
  std::vector<char const *> tmp;
  std::copy_n(argv_in, argc_in, std::back_inserter(tmp));
  return tmp;
}

struct ConditionalAtoi
{
  ConditionalAtoi(const int argc_in, char * argv_in[]) : argv(args_to_strings(argc_in, argv_in))
  {
  }

  template<typename T>
  void
  operator()(T & prm, const std::size_t index)
  {
    if(argv.size() <= index)
      return;
    if(std::strcmp(argv[index], skipper) == 0)
      return;
    prm = std::atoi(argv[index]);
  }

  std::vector<char const *> argv;
};

struct ConditionalAtof
{
  ConditionalAtof(const int argc_in, char * argv_in[]) : argv(args_to_strings(argc_in, argv_in))
  {
  }

  template<typename T>
  void
  operator()(T & prm, const std::size_t index)
  {
    if(argv.size() <= index)
      return;
    if(std::strcmp(argv[index], skipper) == 0)
      return;
    prm = std::atof(argv[index]);
  }

  std::vector<char const *> argv;
};



constexpr unsigned long long
pow(const unsigned int base, const int iexp)
{
  // The "exponentiation by squaring" algorithm used below has to be
  // compressed to one statement due to C++11's restrictions on constexpr
  // functions. A more descriptive version would be:
  //
  // <code>
  // if (iexp <= 0)
  //   return 1;
  //
  // // if the current exponent is not divisible by two,
  // // we need to account for that.
  // const unsigned int prefactor = (iexp % 2 == 1) ? base : 1;
  //
  // // a^b = (a*a)^(b/2)      for b even
  // // a^b = a*(a*a)^((b-1)/2 for b odd
  // return prefactor * ::Utilities::pow(base*base, iexp/2);
  // </code>

  return iexp <= 0 ? 1 : (((iexp % 2 == 1) ? base : 1) * ::Util::pow(base * base, iexp / 2));
}



std::string
si_metric_prefix(unsigned long long measurement)
{
  std::array<std::string, 8> prefixes = {"", "k", "M", "G", "T", "P", "E", "Z"};
  std::ostringstream         oss;

  constexpr std::size_t base = 1000;
  std::size_t           iexp = 0;
  unsigned long long    div  = measurement;
  while(!(div < 1000))
    div = measurement / Util::pow(base, ++iexp);

  oss << div << prefixes[iexp];
  return oss.str();
}



std::string
damping_to_fstring(double factor)
{
  std::ostringstream oss;
  oss << factor;
  return oss.str();
}



std::string
short_name(const std::string & str_in)
{
  std::string sname = str_in.substr(0, 4);
  std::transform(sname.begin(), sname.end(), sname.begin(), [](auto c) { return std::tolower(c); });
  return sname;
}



template<typename MatrixType,
         typename VectorType = LinearAlgebra::distributed::Vector<typename MatrixType::value_type>>
struct MatrixWrapper
{
  using value_type  = typename MatrixType::value_type;
  using vector_type = VectorType;

  MatrixWrapper(const MatrixType & matrix_in) : matrix(matrix_in)
  {
  }

  types::global_dof_index
  m() const
  {
    return matrix.m();
  }

  types::global_dof_index
  n() const
  {
    return matrix.n();
  }

  void
  vmult(const ArrayView<value_type> dst_view, const ArrayView<const value_type> src_view) const
  {
    AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1U,
                ExcMessage("No MPI support"));
    vector_type dst(dst_view.size());
    vector_type src(src_view.size());

    std::copy(src_view.cbegin(), src_view.cend(), src.begin());
    matrix.vmult(dst, src);
    std::copy(dst.begin(), dst.end(), dst_view.begin());
  }

  FullMatrix<value_type>
  as_fullmatrix()
  {
    return table_to_fullmatrix(Tensors::matrix_to_table(*this));
  }

  const MatrixType & matrix;
};

/**
   * Point-wise comparator for renumbering global dofs
   * from hierarchy to lexicographic ordering.
   * @tparam dim dimension
   */
  template <int dim>
  struct ComparePointwiseLexicographic;

  template <>
  struct ComparePointwiseLexicographic<1>
  {
    ComparePointwiseLexicographic() = default;
    bool
    operator()(const std::pair<Point<1>, types::global_dof_index> &c1,
               const std::pair<Point<1>, types::global_dof_index> &c2) const
    {
      return c1.first[0] < c2.first[0];
    }
  };

  template <>
  struct ComparePointwiseLexicographic<2>
  {
    ComparePointwiseLexicographic() = default;
    bool
    operator()(const std::pair<Point<2>, types::global_dof_index> &c1,
               const std::pair<Point<2>, types::global_dof_index> &c2) const
    {
      const double y_err = std::abs(c1.first[1] - c2.first[1]);

      if (y_err > 1e-10 && c1.first[1] < c2.first[1])
        return true;
      // y0 == y1
      if (y_err < 1e-10 && c1.first[0] < c2.first[0])
        return true;
      return false;
    }
  };

  template <>
  struct ComparePointwiseLexicographic<3>
  {
    ComparePointwiseLexicographic() = default;
    bool
    operator()(const std::pair<Point<3>, types::global_dof_index> &c1,
               const std::pair<Point<3>, types::global_dof_index> &c2) const
    {
      const double z_err = std::abs(c1.first[2] - c2.first[2]);
      const double y_err = std::abs(c1.first[1] - c2.first[1]);

      if (z_err > 1e-10 && c1.first[2] < c2.first[2])
        return true;
      // z0 == z1
      if (z_err < 1e-10 && y_err > 1e-10 && c1.first[1] < c2.first[1])
        return true;
      // z0 == z1, y0 == y1
      if (z_err < 1e-10 && y_err < 1e-10 && c1.first[0] < c2.first[0])
        return true;

      return false;
    }
  };

  /**
   * Compute the set of renumbering indices on finest level needed by the
   * Lexicographic() function. Does not perform the renumbering on the
   * DoFHandler dofs but returns the renumbering vector.
   *
   * Using @b DoFHandler<dim, spacedim>::active_cell_iterators loop all cells.
   * @tparam dim
   * @tparam spacedim
   * @param new_indices
   * @param dof
   */
  template <int dim, int spacedim>
  void
  compute_Lexicographic(std::vector<types::global_dof_index> &new_indices,
                        const DoFHandler<dim, spacedim>      &dof)
  {
    // Assert((dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
    //           &dof.get_triangulation()) == nullptr),
    //        ExcNotImplemented());

    const unsigned int                                    n_dofs = dof.n_dofs();
    std::vector<std::pair<Point<spacedim>, unsigned int>> support_point_list(
      n_dofs);

    const hp::FECollection<dim> &fe_collection = dof.get_fe_collection();
    Assert(fe_collection[0].has_support_points(),
           typename FiniteElement<dim>::ExcFEHasNoSupportPoints());
    hp::QCollection<dim> quadrature_collection;
    for (unsigned int comp = 0; comp < fe_collection.size(); ++comp)
      {
        Assert(fe_collection[comp].has_support_points(),
               typename FiniteElement<dim>::ExcFEHasNoSupportPoints());
        quadrature_collection.push_back(
          Quadrature<dim>(fe_collection[comp].get_unit_support_points()));
      }
    hp::FEValues<dim, spacedim> hp_fe_values(fe_collection,
                                             quadrature_collection,
                                             update_quadrature_points);

    std::vector<bool> already_touched(n_dofs, false);

    std::vector<types::global_dof_index> local_dof_indices;

    for (const auto &cell : dof.active_cell_iterators())
      {
        const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
        local_dof_indices.resize(dofs_per_cell);
        hp_fe_values.reinit(cell);
        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
        cell->get_active_or_mg_dof_indices(local_dof_indices);
        const std::vector<Point<spacedim>> &points =
          fe_values.get_quadrature_points();
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          if (!already_touched[local_dof_indices[i]])
            {
              support_point_list[local_dof_indices[i]].first = points[i];
              support_point_list[local_dof_indices[i]].second =
                local_dof_indices[i];
              already_touched[local_dof_indices[i]] = true;
            }
      }

    ComparePointwiseLexicographic<spacedim> comparator;
    std::sort(support_point_list.begin(), support_point_list.end(), comparator);
    for (types::global_dof_index i = 0; i < n_dofs; ++i)
      new_indices[support_point_list[i].second] = i;
  }
  /**
   * Compute the set of renumbering indices on one level of a multigrid
   * hierarchy needed by the Lexicographic() function. Does not perform the
   * renumbering on the DoFHandler dofs but returns the renumbering vector.
   *
   * Using @b DoFHandler<dim, spacedim>::level_cell_iterator loop all cells.
   * @tparam dim
   * @tparam spacedim
   * @param new_indices
   * @param dof
   * @param level
   */
  template <int dim, int spacedim>
  void
  compute_Lexicographic(std::vector<types::global_dof_index> &new_indices,
                        const DoFHandler<dim, spacedim>      &dof,
                        const unsigned int                    level)
  {
    Assert(dof.get_fe().has_support_points(),
           typename FiniteElement<dim>::ExcFEHasNoSupportPoints());
    const unsigned int n_dofs = dof.n_dofs(level);
    std::vector<std::pair<Point<spacedim>, unsigned int>> support_point_list(
      n_dofs);

    Quadrature<dim>         q_dummy(dof.get_fe().get_unit_support_points());
    FEValues<dim, spacedim> fe_values(dof.get_fe(),
                                      q_dummy,
                                      update_quadrature_points);

    std::vector<bool> already_touched(dof.n_dofs(), false);

    const unsigned int dofs_per_cell = dof.get_fe().n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    typename DoFHandler<dim, spacedim>::level_cell_iterator begin =
      dof.begin(level);
    typename DoFHandler<dim, spacedim>::level_cell_iterator end =
      dof.end(level);
    for (; begin != end; ++begin)
      {
        const typename Triangulation<dim, spacedim>::cell_iterator &begin_tria =
          begin;
        begin->get_active_or_mg_dof_indices(local_dof_indices);
        fe_values.reinit(begin_tria);
        const std::vector<Point<spacedim>> &points =
          fe_values.get_quadrature_points();
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          if (!already_touched[local_dof_indices[i]])
            {
              support_point_list[local_dof_indices[i]].first = points[i];
              support_point_list[local_dof_indices[i]].second =
                local_dof_indices[i];
              already_touched[local_dof_indices[i]] = true;
            }
      }
    ComparePointwiseLexicographic<spacedim> comparator;
    std::sort(support_point_list.begin(), support_point_list.end(), comparator);
    for (types::global_dof_index i = 0; i < n_dofs; ++i)
      new_indices[support_point_list[i].second] = i;
  }

  /**
   * Lexicographic numbering on finest level.
   * @tparam dim
   * @tparam spacedim
   * @param dof
   */
  template <int dim, int spacedim>
  void
  Lexicographic(DoFHandler<dim, spacedim> &dof)
  {
    std::vector<types::global_dof_index> renumbering(dof.n_dofs());
    compute_Lexicographic(renumbering, dof);
    dof.renumber_dofs(renumbering);
  }
  /**
   * Lexicographic numbering on one level of a multigrid hierarchy.
   * @tparam dim
   * @tparam spacedim
   * @param dof
   * @param level
   */
  template <int dim, int spacedim>
  void
  Lexicographic(DoFHandler<dim, spacedim> &dof, const unsigned int level)
  {
    std::vector<types::global_dof_index> renumbering(dof.n_dofs(level));
    compute_Lexicographic(renumbering, dof, level);
    dof.renumber_dofs(level, renumbering);
  }

} // end namespace Util

#endif /* UTILITIES_H_ */
