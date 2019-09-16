#ifndef MAPPINGINFO_H
#define MAPPINGINFO_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/hp/q_collection.h>

#include <array>
#include <memory>

#include "TPSS.h"
#include "patch_info.h"
#include "tensors.h"

namespace TPSS
{
template<int dim, typename number>
class MappingInfo
{
public:
  using CellIterator               = typename dealii::DoFHandler<dim>::level_cell_iterator;
  using GeometryType               = dealii::internal::MatrixFreeFunctions::GeometryType;
  using PatchType                  = GeometryType;
  using VectorizedArrayType        = dealii::VectorizedArray<number>;
  static constexpr auto macro_size = dealii::VectorizedArray<number>::n_array_elements;

  struct AdditionalData;

  struct InternalData;

  struct LocalData;

  void
  initialize_storage(const PatchInfo<dim> &                 patch_info,
                     const MatrixFreeConnect<dim, number> & mf_connect,
                     const AdditionalData &                 additional_data);

  void
  clear();

  const dealii::VectorizedArray<number> *
  h_inverses_begin(const unsigned int patch_id) const;

  const dealii::VectorizedArray<number> *
  h_lengths_begin(const unsigned int patch_id) const;

  bool mapping_data_initialized = false;

private:
  LocalData
  extract_cartesian_scaling(const std::array<unsigned int, 3> * const     batch_triple,
                            const std::pair<unsigned int, unsigned int> * bid_count_pair,
                            const unsigned int                            n_batches) const;

  LocalData
  compute_average_scaling(dealii::FEValues<dim> &          fe_values,
                          const PatchWorker<dim, number> & patch_worker,
                          const unsigned int               patch_id) const;

  void
  submit_local_data(const LocalData & local_data);

  // check whether the Jacobian is a scaled identity matrix to machine prec
  bool
  is_uniform_cell(const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> * jacobian,
                  const double zero_tolerance_double) const;

  // check whether the two Jacobians equal each other up to machine prec
  bool
  are_jacobians_equal(const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> * jacobian0,
                      const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> * jacobian,
                      const double zero_tolerance_double) const;

  const dealii::MatrixFree<dim, number> * mf_storage;
  const dealii::internal::MatrixFreeFunctions::MappingInfo<dim, number, VectorizedArrayType> *
    mf_mapping_info = nullptr;
  const dealii::internal::MatrixFreeFunctions::
    MappingInfoStorage<dim, dim, number, VectorizedArrayType> * mf_mapping_data = nullptr;

  unsigned int n_subdomains          = static_cast<unsigned int>(-1);
  int          patch_size            = -1;
  int          n_cells_per_direction = -1;

  std::vector<unsigned int> mapping_data_starts;

  AdditionalData additional_data;

  InternalData internal_data;

  // // Helper array used during initialization
  // mutable std::vector<bool> batch_is_uniform;
};

template<int dim, typename number>
struct MappingInfo<dim, number>::AdditionalData
{
  unsigned int n_q_points      = 5;
  bool         normalize_patch = false;
  bool         use_arc_length  = true;
};

template<int dim, typename number>
struct MappingInfo<dim, number>::InternalData
{
  InternalData() = default;

  InternalData(const InternalData &) = delete;

  ~InternalData()
  {
    clear();
  }

  InternalData &
  operator=(const InternalData &) = delete;

  void
  clear();

  /**
   * lexicographical order: cell_no_1d < direction < patch_id
   */
  dealii::AlignedVector<dealii::VectorizedArray<number>> h_inverses;

  /**
   * lexicographical order: cell_no_1d < direction < patch_id
   */
  dealii::AlignedVector<dealii::VectorizedArray<number>> h_lengths;
};

template<int dim, typename number>
struct MappingInfo<dim, number>::LocalData
{
  LocalData(int cells_per_direction)
  {
    for(auto & h : h_inverses)
      h.resize_fast(static_cast<std::size_t>(cells_per_direction));
  }

  LocalData(const LocalData &) = default;

  ~LocalData() = default;

  LocalData &
  operator=(const LocalData &) = default;

  // bool compare_scaling (const LocalData& other, const double zero_tolerance)
  // {
  //   for (unsigned int direction = 0; direction < dim; ++direction)
  //     for (unsigned int cell = 0; cell < h_inverses[direction].size(); ++cell)
  // 	for (unsigned int vv = 0; vv < dealii::VectorizedArray<number>::n_array_elements; ++vv)
  // 	  if (std::fabs (h_inverses[direction][cell][vv] - other.h_inverses[direction][cell][vv]) >
  // zero_tolerance) 	    return false;
  //   return true;
  // }

  std::array<dealii::AlignedVector<dealii::VectorizedArray<number>>, dim> h_inverses;
};

//  ++++++++++++++++++++++++++++++   inline functions   ++++++++++++++++++++++++++++++

template<int dim, typename number>
void
MappingInfo<dim, number>::clear()
{
  mf_storage      = nullptr;
  mf_mapping_info = nullptr;
  mf_mapping_data = nullptr;

  n_subdomains          = static_cast<unsigned int>(-1);
  patch_size            = -1;
  n_cells_per_direction = -1;

  mapping_data_starts.clear();
  additional_data = AdditionalData{};
  internal_data.clear();
}

template<int dim, typename number>
const dealii::VectorizedArray<number> *
MappingInfo<dim, number>::h_lengths_begin(const unsigned int patch_id) const
{
  Assert(mapping_data_initialized, dealii::ExcNotInitialized());
  AssertIndexRange(patch_id, n_subdomains);
  return internal_data.h_lengths.begin() + mapping_data_starts[patch_id];
}

template<int dim, typename number>
const dealii::VectorizedArray<number> *
MappingInfo<dim, number>::h_inverses_begin(const unsigned int patch_id) const
{
  Assert(mapping_data_initialized, dealii::ExcNotInitialized());
  AssertIndexRange(patch_id, n_subdomains);
  return internal_data.h_inverses.begin() + mapping_data_starts[patch_id];
}

template<int dim, typename number>
inline bool
MappingInfo<dim, number>::is_uniform_cell(
  const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> * jacobian,
  const double                                                    zero_tolerance_double) const
{
  bool         is_uniform = true;
  const auto & jac_0      = jacobian[0];
  for(unsigned int vv = 0; vv < macro_size; ++vv)
    for(unsigned int d = 1; d < dim; ++d)
      if(std::fabs(jac_0[0][0][vv] - jac_0[d][d][vv]) > zero_tolerance_double)
      {
        is_uniform = false;
        //	  std::cout << jac_0[0][0][vv] << "-" << jac_0[d][d][vv] << std::endl ;
        break;
      }
  return is_uniform;
}

template<int dim, typename number>
inline bool
MappingInfo<dim, number>::are_jacobians_equal(
  const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> * jacobian0,
  const dealii::Tensor<2, dim, dealii::VectorizedArray<number>> * jacobian,
  const double                                                    zero_tolerance_double) const
{
  bool         are_equal = true;
  const auto & jac_0     = jacobian0[0];
  const auto & jac       = jacobian[0];
  for(unsigned int vv = 0; vv < macro_size; ++vv)
    for(unsigned int d1 = 0; d1 < dim; ++d1)
      for(unsigned int d2 = 0; d2 < dim; ++d2)
        if(std::fabs(jac_0[d1][d2][vv] - jac[d1][d2][vv]) > zero_tolerance_double)
        {
          are_equal = false;
          //	  std::cout << jac_0[0][0][vv] << "-" << jac_0[d][d][vv] << std::endl ;
          break;
        }
  return are_equal;
}

template<int dim, typename number>
inline void
MappingInfo<dim, number>::submit_local_data(const MappingInfo<dim, number>::LocalData & local_data)
{
  const unsigned int n_cells_per_direction = local_data.h_inverses[0].size();
  for(unsigned int d = 0; d < dim; ++d)
    for(unsigned int cell_no_1d = 0; cell_no_1d < n_cells_per_direction; ++cell_no_1d)
    {
      internal_data.h_inverses.push_back(local_data.h_inverses[d][cell_no_1d]);
      internal_data.h_lengths.push_back(1. / local_data.h_inverses[d][cell_no_1d]);
    }
  AssertDimension(internal_data.h_inverses.size(), internal_data.h_lengths.size());
}

// --------------------------------   InternalData   --------------------------------

template<int dim, typename number>
void
MappingInfo<dim, number>::InternalData::clear()
{
  h_inverses.clear();
  h_lengths.clear();
}

} // end namespace TPSS

#include "mapping_info.templates.h"

#endif // end inclusion guard
