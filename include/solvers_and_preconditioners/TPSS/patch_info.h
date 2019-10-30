#ifndef PATCHINFO_H
#define PATCHINFO_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/graph_coloring.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "TPSS.h"
#include "generic_functionalities.h"
#include "time_info.h"

#include <array>
#include <fstream>
#include <memory>

namespace TPSS
{
// TODO patch_info should contain TriaIterators to TriaAccessors not DoFAccessor ?!
template<int dim>
class PatchInfo
{
public:
  using CellIterator  = typename dealii::DoFHandler<dim>::level_cell_iterator;
  using PatchIterator = typename std::vector<std::vector<CellIterator>>::const_iterator;

  struct AdditionalData;

  struct PartitionData;

  struct InternalData;

  PatchInfo() = default;

  PatchInfo(const PatchInfo<dim> &) = delete;

  ~PatchInfo()
  {
    clear();
  }

  PatchInfo<dim> &
  operator=(const PatchInfo<dim> &) = delete;

  void
  initialize(const dealii::DoFHandler<dim> * dof_handler, const AdditionalData additional_data);

  void
  clear();

  bool
  empty() const
  {
    return internal_data.empty();
  }

  /**
   * Read access to the internal data.
   */
  const InternalData *
  get_internal_data() const;

  /**
   * Read access to the additional data
   */
  const AdditionalData &
  get_additional_data() const;

  /**
   * An array of strides to access the CellIterator collections
   * (=patch) within the flat array @p cell_iterators. In other words
   * two consecutive elements in @p patch_starts represent a half-open
   * range of CellIterators, containing all cells to construct a
   * (macro) patch.
   */
  std::vector<unsigned int> patch_starts;

  /**
   * An array that stores how many vectorization lanes of a macro patch contain meaningful data
   */
  std::vector<unsigned int> n_lanes_filled;

  /**
   * A flat array that stores the information if each face within a
   * (macro) face is at the physical boundary. Faces are ordered in
   * the standard deal.II way, see GeometryInfo. The integer must be
   * reinterpreted as std::bitset of length @p macro_size.
   * Lexicographical:   face number   <   patch id
   */
  // TODO instead on patch-level we need masks on cell-level ... (not
  // generic!!)  TODO replace this field. we can extract the
  // information given the cell collections, i.e
  // std::vector<CellIterators>
  std::vector<unsigned short> at_boundary_mask;

  /**
   * Container storing the partitioning into colored batches of
   * interior/boundary, incomplete/complete subdomains.
   */
  PartitionData subdomain_partition_data;

  /**
   * An array to store timer output.
   */
  std::vector<TimeInfo> time_data;

  struct GhostPatch
  {
    GhostPatch(const unsigned int proc, const CellId & cell_id)
    {
      submit_id(proc, cell_id);
    }

    void
    submit_id(const unsigned int proc, const CellId & cell_id)
    {
      const auto member = proc_to_cell_ids.find(proc);
      if(member != proc_to_cell_ids.cend())
      {
        member->second.emplace_back(cell_id);
        Assert(!(member->second.empty()), ExcMessage("at least one element"));
      }
      else
      {
        const auto status = proc_to_cell_ids.emplace(proc, std::vector<CellId>{cell_id});
        (void)status;
        Assert(status.second, ExcMessage("failed to insert key-value-pair"));
      }
    }

    std::string
    str() const
    {
      std::ostringstream oss;
      oss << "{";
      const auto size = proc_to_cell_ids.size();
      unsigned   i    = 0;
      for(auto key_value = proc_to_cell_ids.cbegin(); key_value != proc_to_cell_ids.cend();
          ++key_value, ++i)
        oss << "(" << key_value->first << ", " << vector_to_string(key_value->second)
            << ((i + 1) < size ? "), " : ")}");
      return oss.str();
    }

    std::map<unsigned, std::vector<CellId>> proc_to_cell_ids;
  };

private:
  static std::vector<types::global_dof_index>
  get_face_conflicts(const PatchIterator & patch)
  {
    std::vector<types::global_dof_index> conflicts;
    const auto &                         cell_collection = *patch;

    for(const auto & cell : cell_collection)
      for(unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
      {
        const bool neighbor_has_same_level = (cell->neighbor_level(face_no) == cell->level());
        const bool neighbor_doesnt_exist   = (cell->neighbor_level(face_no) == -1);
        const bool non_adaptive            = neighbor_has_same_level || neighbor_doesnt_exist;
        (void)non_adaptive;
        Assert(non_adaptive, ExcNotImplemented());
        conflicts.emplace_back(cell->face(face_no)->index());
      }
    return conflicts;
  }

  void
  initialize_cell_patches(const dealii::DoFHandler<dim> * dof_handler,
                          const AdditionalData            additional_data);

  /**
   * Gathering the locally owned and ghost cells attached to a common
   * vertex as the collection of cell iterators (patch). The
   * successive distribution of the collections with ghost cells
   * currently follows the logic:
   *
   * 1.) owns one mpi-proc more than half of the cells (locally owned)
   * of the vertex patch the mpi-proc takes the ownership
   *
   * 2.) for the remaining ghost patches the mpi-proc with the cell of
   * the lowest CellId (see dealii::Triangulation) takes the ownership
   */
  std::vector<std::vector<CellIterator>>
  gather_vertex_patches(const DoFHandler<dim> & dof_handler,
                        const AdditionalData &  additional_data) const;

  void
  initialize_vertex_patches(const dealii::DoFHandler<dim> * dof_handler,
                            const AdditionalData            additional_data);

  // TODO access by public method
  std::vector<std::pair<unsigned int, unsigned int>>
  reorder_colors(const std::vector<
                 std::vector<typename std::vector<std::vector<CellIterator>>::const_iterator>> &
                   colored_cells) const;

  // TODO access by public print function
  void
  write_visual_data(
    const dealii::DoFHandler<dim> &                            dof_handler,
    const std::vector<std::pair<unsigned int, unsigned int>> & reordered_colors) const;

  template<int size_reg>
  void
  submit_patches(const std::vector<PatchIterator> & patch_iterators)
  {
    // *** submit the interior subdomains first
    unsigned int               n_interior_subdomains_reg = 0;
    std::vector<PatchIterator> boundary_patch_reg;
    for(auto patch : patch_iterators)
    {
      const bool patch_at_boundary =
        std::any_of(patch->cbegin(), patch->cend(), IteratorFilters::AtBoundary{});
      if(patch->size() == size_reg) // regular
      {
        if(patch_at_boundary)
        {
          boundary_patch_reg.push_back(patch);
        }
        else
        {
          ++n_interior_subdomains_reg;
          for(const auto & cell : *patch)
            internal_data.cell_iterators.emplace_back(cell);
        }
      }
      else // irregular
        Assert(false, ExcNotImplemented());
    }

    // *** submit the boundary subdomains next
    for(const auto it : boundary_patch_reg)
    {
      for(const auto & cell : *it)
        internal_data.cell_iterators.emplace_back(cell);
    }
    internal_data.n_interior_subdomains.emplace_back(n_interior_subdomains_reg);
    internal_data.n_boundary_subdomains.emplace_back(boundary_patch_reg.size());
  }

  ConditionalOStream pcout{std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0};

  /**
   * Storing the actual data shaping patches.
   */
  InternalData   internal_data;
  AdditionalData additional_data;
};

template<int dim>
struct PatchInfo<dim>::AdditionalData
{
  TPSS::PatchVariant    patch_variant    = TPSS::PatchVariant::invalid;
  TPSS::SmootherVariant smoother_variant = TPSS::SmootherVariant::invalid;
  unsigned int          level            = -1;
  std::function<
    std::vector<std::vector<PatchIterator>>(const std::vector<std::vector<CellIterator>> & patches,
                                            const AdditionalData additional_data)>
    coloring_func;
  std::function<void(const DoFHandler<dim> *                  dof_handler,
                     const AdditionalData                     additional_data,
                     std::vector<std::vector<CellIterator>> & cell_collections)>
    manual_gathering_func;
  std::function<void(const DoFHandler<dim> & dof_handler_in,
                     const std::vector<std::vector<typename TPSS::PatchInfo<dim>::PatchIterator>> &
                       colored_iterators,
                     const std::string)>
       visualize_coloring;
  bool print_details = false; // DEBUG
};

/**
 * This helper struct stores the information on the distribution of
 * CellIterator collections (=patch) into partitions of the same
 * constitution for each color. In other words, patch properties are
 * uniquely determined by the half-open patch range (namely a
 * partition).
 *
 * It is an abstract container providing access and information of the
 * distribution. The actual distribution and assignment of the patch
 * properties is treated by worker classes, see PatchWorker. Moreover,
 * the actual CellIterator storage is encapsulated in PatchInfo's @p
 * internal_data.
 */
template<int dim>
struct PatchInfo<dim>::PartitionData
{
  PartitionData() = default;

  PartitionData(const PartitionData &) = default;

  ~PartitionData()
  {
    clear();
  };

  PartitionData &
  operator=(const PartitionData &) = default;

  void
  clear();

  std::size_t
  n_colors() const;

  std::size_t
  n_partitions(const unsigned int color = 0) const;

  std::size_t
  n_subdomains() const;

  std::size_t
  n_subdomains(const unsigned int color) const;

  /**
   * Returns the total, half-open patch range including all colors.
   */
  std::pair<unsigned int, unsigned int>
  get_patch_range() const;

  /**
   * Returns the half-open patch range associated to the color @p
   * color.
   */
  std::pair<unsigned int, unsigned int>
  get_patch_range(const unsigned int color) const;

  /**
   * Returns the half-open patch range of partition @p partition
   * regarding the color @p color.
   */
  std::pair<unsigned int, unsigned int>
  get_patch_range(const unsigned int partition, const unsigned int color) const;

  /**
   * Checks the compatibility of @p other against the current
   * partition data. This identifies if data might be reused.
   */
  bool
  check_compatibility(const PartitionData & other) const;

  /**
   * Actual storage of a set of partitions for each color. Two
   * consecutive integers of the inner vector represent a half-open
   * interval, the so-called patch range identified by a partition id.
   */
  std::vector<std::vector<unsigned int>> partitions;
};

/**
 * This helper struct contains all data to construct (macro) patches
 * and the distribution into partitions of certain properties for each
 * color. Up to now it only contains data for regular patches to
 * obtain tensor product structures.
 * A (macro) patch is characterized as the contiguous range of a starting
 * and ending CellIterator in the flat array @p cell_iterators.
 */
template<int dim>
struct PatchInfo<dim>::InternalData
{
  InternalData() = default;

  InternalData(const InternalData &) = delete;

  ~InternalData()
  {
    clear();
  };

  InternalData &
  operator=(const InternalData &) = delete;

  void
  clear();

  bool
  empty() const
  {
    return cell_iterators.empty();
  }

  bool
  empty_on_all() const
  {
    const auto n_iterators_mpimax = Utilities::MPI::max(cell_iterators.size(), MPI_COMM_WORLD);
    return (n_iterators_mpimax == 0);
  }

  unsigned int level = -1;

  /**
   * Number of interior patches for each color.
   */
  std::vector<unsigned int> n_interior_subdomains;

  /**
   * Number of boundary patches for each color.
   */
  std::vector<unsigned int> n_boundary_subdomains;

  /**
   * Flat array that stores all CellIterators for the construction of
   * (macro) patches. The successive alignment of CellIterators is
   * such that all interior subdomains are stored first, then the ones
   * at the physical boundary
   *
   * Lexicographical:  cell number  <  lane  <  patch id  <   color
   */
  std::vector<CellIterator> cell_iterators;
};

template<int dim, typename number>
struct MatrixFreeConnect
{
  /**
   * For a given macro patch, given by @p patch_id, the number of batches
   * @p n_batches (not trivial, if there exist cells in batch that occur in more
   * than one micro patch, or vice versa) is returned. The iterator range of (bid, count)-pairs
   * is then given by [@p id_count_pair, @p id_count_pair + @p n_batches). Moreover,
   * the starting triple @p triple is set.
   */
  unsigned int
  set_pointers_and_count(const unsigned int                             patch_id,
                         const std::array<unsigned int, 3> *&           triple,
                         const std::pair<unsigned int, unsigned int> *& id_count_pair) const;
  /**
   * The underlying MatrixFree object used to map matrix-free infrastructure
   * to the patch distribution stored in PatchInfo
   */
  const dealii::MatrixFree<dim, number> * mf_storage = nullptr;

  /**
   * In the MatrixFree framework cells are clustered into so-called (cell-)batches
   * due to vectorization. Every batch is associated to an integer, similar to the
   * integer-identification of vectorized patches by means of the @p subdomain_partition_data
   * Filled and used by the PatchWorker this vector in conjunction with @p batch_count_per_id
   * and @p batch_starts provides a unique mapping between cells for each
   * patch (PatchWorker) and cells within batches (MatrixFree).
   * Given a macro patch, represented by the integer @p patch_id, the half-open range
   * [@p batch_starts[patch_id], @p batch_starts[patch_id+1]) identifies a range of pairs,
   * stored in @p batch_count_per_id, to
   * shape the macro patch associated to @patch_id. It happens that some of those batches
   * contain more than one cell. This information is stored in the @p batch_count_per_id
   * pairs, with the batch_id being the first and the corresponding count of cells the second
   * member. Each pair is associated to @p count triples in @p bcomp_vcomp_cindex. The first member
   * of the triple is the vectorization component within a batch, the second is the vectorization
   * component within a patch and the third is the local cell index (lexicographically ordered)
   * within a patch. The flat data field @p bcomp_vcomp_cindex is strided through by the @p patch_id
   * with steps of size (@p patch_size * @p vectorization_length).
   */
  std::vector<std::array<unsigned int, 3>> bcomp_vcomp_cindex;

  /**
   * For details see bcomp_vcomp_cindex.
   */
  std::vector<std::pair<unsigned int, unsigned int>> batch_count_per_id;

  /**
   * For details see bcomp_vcomp_cindex.
   */
  std::vector<std::size_t> batch_starts;

  /**
   * Sets the stride with respect to @p bcomp_vcomp_cindex.
   */
  int stride_triple = -1;
};

/**
 * A worker class re-interpreting the raw patch data in PatchInfo with
 * respect to a vectorized structure. The new alignment is uniquely
 * determined by the PartitionData @p subdomain_partition_data and @p
 * patch_starts stored in the PatchInfo. In addition, we extract
 * physical informations on subdomains and store them in the
 * PatchInfo, e.g. @p at_boundary_mask fields.
 *
 * Partitions per color are set as follows:
 * 1st partition = interior incomplete
 * 2nd partition = boundary incomplete
 * 3rd partition = interior complete
 * 4th partition = boundary complete
 *
 * An incomplete macro patch is a duplication of a physical patch
 * along all vectorization lanes. In other words incomplete patches
 * occur when the amount of physical patches is not divisible by the
 * vectorization length.  An interior macro patch contains several
 * interior patches.
 */
template<int dim, typename number>
class PatchWorker
{
public:
  static constexpr unsigned int macro_size = dealii::VectorizedArray<number>::n_array_elements;
  using CellIterator                       = typename PatchInfo<dim>::CellIterator;

  PatchWorker() = delete;

  PatchWorker(PatchInfo<dim> & patch_info);

  PatchWorker(const PatchInfo<dim> & patch_info);

  PatchWorker(const PatchWorker &) = delete;

  PatchWorker &
  operator=(const PatchWorker &) = delete;

  void
  clear_mf_connect(MatrixFreeConnect<dim, number> & mf_connect);

  /**
   * We fill the MatrixFreeConnect class which creates a link between the
   * MatrixFree infrastructure and the vectorized patch distribution, that has
   * been initialized by this class.
   */
  void
  connect_to_matrixfree(MatrixFreeConnect<dim, number> & mf_connect);

  /**
   * Based on the data on the flat and unvectorized patch distribution
   * in PatchInfo::InternalData we are able to partition subdomains
   * into predicated groups.
   */
  static void
  compute_partition_data(typename PatchInfo<dim>::PartitionData &      partition_data,
                         const typename PatchInfo<dim>::InternalData * internal_data);

  unsigned int
  n_physical_subdomains() const;

  unsigned int
  n_lanes_filled(const unsigned int patch_id) const;

  /**
   * Returns the collection of macro cells describing the macro patch
   * @p patch_id subject to a lexicographical ordering.
   */
  std::vector<std::array<CellIterator, macro_size>>
  get_cell_collection(unsigned int patch_id) const;

  std::vector<ArrayView<const CellIterator>>
  get_cell_collection_views(unsigned int patch_id) const;

  const typename PatchInfo<dim>::PartitionData &
  get_partition_data() const;

private:
  // void
  // initialize(const PatchInfo<dim> & patch_info);

  void
  clear_patch_info(PatchInfo<dim> & info);

  /**
   * This method partitions the (unvectorized) patches, contained in
   * PatchInfo, into interior/boundary, incomplete/complete groups,
   * containing vectorized patches.
   */
  void
  partition_patches(PatchInfo<dim> & patch_info);

  /**
   * Checks whether the partitions for the given color are valid.
   */
  static void
  has_valid_state(const typename PatchInfo<dim>::PartitionData & subdomain_partition_data,
                  const unsigned int                             color);

  const PatchInfo<dim> * const patch_info;
  const unsigned int           patch_size = 0;
};

//  ++++++++++++++++++++++++++++++   inline functions   ++++++++++++++++++++++++++++++

// --------------------------------   PatchInfo   --------------------------------

template<int dim>
inline void
PatchInfo<dim>::clear()
{
  patch_starts.clear();
  at_boundary_mask.clear();
  subdomain_partition_data.clear();

  internal_data.clear();
  additional_data = AdditionalData{};
  time_data.clear();
}

template<int dim>
inline const typename PatchInfo<dim>::InternalData *
PatchInfo<dim>::get_internal_data() const
{
  return &internal_data;
}

template<int dim>
inline const typename PatchInfo<dim>::AdditionalData &
PatchInfo<dim>::get_additional_data() const
{
  return additional_data;
}

// --------------------------------   PatchInfo::InternalData   --------------------------------

template<int dim>
inline void
PatchInfo<dim>::InternalData::clear()
{
  level = -1;
  n_interior_subdomains.clear();
  n_boundary_subdomains.clear();
  cell_iterators.clear();
}

// --------------------------------   PatchInfo::PartitionData   --------------------------------

template<int dim>
inline void
PatchInfo<dim>::PartitionData::clear()
{
  partitions.clear();
}

template<int dim>
inline std::size_t
PatchInfo<dim>::PartitionData::n_colors() const
{
  return partitions.size();
}

template<int dim>
inline std::size_t
PatchInfo<dim>::PartitionData::n_partitions(const unsigned int color) const
{
  AssertIndexRange(color, partitions.size());
  return (partitions[color].size() > 0) ? partitions[color].size() - 1 : 0;
}

template<int dim>
inline std::size_t
PatchInfo<dim>::PartitionData::n_subdomains() const
{
  // Assert(!partitions.empty(), ExcMessage("TODO should not be empty."));
  return partitions.empty() ? 0 : (partitions.back().empty()) ? 0 : partitions.back().back();
}

template<int dim>
inline std::size_t
PatchInfo<dim>::PartitionData::n_subdomains(const unsigned int color) const
{
  AssertIndexRange(color, partitions.size());
  return partitions[color].empty() ? 0 : partitions[color].back();
}

template<int dim>
inline std::pair<unsigned int, unsigned int>
PatchInfo<dim>::PartitionData::get_patch_range() const
{
  Assert(!partitions.empty(), dealii::ExcInvalidState());
  Assert(!partitions.front().empty(), dealii::ExcInvalidState());
  Assert(!partitions.back().empty(), dealii::ExcInvalidState());
  return std::make_pair(partitions.front().front(), partitions.back().back());
}

template<int dim>
inline std::pair<unsigned int, unsigned int>
PatchInfo<dim>::PartitionData::get_patch_range(const unsigned int color) const
{
  AssertIndexRange(color, partitions.size());
  Assert(!partitions[color].empty(), dealii::ExcInvalidState());
  return std::make_pair(partitions[color].front(), partitions[color].back());
}

template<int dim>
inline std::pair<unsigned int, unsigned int>
PatchInfo<dim>::PartitionData::get_patch_range(const unsigned int partition,
                                               const unsigned int color) const
{
  AssertIndexRange(color, partitions.size());
  AssertIndexRange(partition, n_partitions());
  return std::make_pair(partitions[color][partition], partitions[color][partition + 1]);
}

template<int dim>
inline bool
PatchInfo<dim>::PartitionData::check_compatibility(const PartitionData & other) const
{
  return (partitions == other.partitions);
}

// --------------------------------   MatrixFreeConnect   --------------------------------

template<int dim, typename number>
inline unsigned int
MatrixFreeConnect<dim, number>::set_pointers_and_count(
  const unsigned int                             patch_id,
  const std::array<unsigned int, 3> *&           triple,
  const std::pair<unsigned int, unsigned int> *& id_count_pair) const
{
  Assert(stride_triple > 0, dealii::ExcNotInitialized());
  AssertDimension(stride_triple % dealii::VectorizedArray<number>::n_array_elements, 0);
  triple        = bcomp_vcomp_cindex.data() + stride_triple * patch_id;
  id_count_pair = batch_count_per_id.data() + batch_starts[patch_id];
  unsigned int n_batches{
    static_cast<unsigned int>(batch_starts[patch_id + 1] - batch_starts[patch_id])};

  const auto &   sum_counts = [](const auto val, const auto & p) { return val + p.second; };
  const unsigned n_triples_accumulated =
    std::accumulate(id_count_pair, id_count_pair + n_batches, 0, sum_counts);
  (void)n_triples_accumulated;
  AssertDimension(stride_triple, n_triples_accumulated);
  return n_batches;
}

// --------------------------------   PatchWorker   --------------------------------

template<int dim, typename number>
PatchWorker<dim, number>::PatchWorker(const PatchInfo<dim> & patch_info_in)
  : patch_info(&patch_info_in),
    patch_size(UniversalInfo<dim>::n_cells(patch_info_in.get_additional_data().patch_variant))
{
  AssertThrow(patch_info_in.get_additional_data().patch_variant != TPSS::PatchVariant::invalid,
              ExcInvalidState());
  typename PatchInfo<dim>::PartitionData subdomain_partition_data;
  compute_partition_data(subdomain_partition_data, patch_info_in.get_internal_data());
  const bool partition_data_is_valid =
    subdomain_partition_data.check_compatibility(patch_info_in.subdomain_partition_data);
  (void)partition_data_is_valid;
  Assert(partition_data_is_valid, ExcMessage("The PartitionData does not fit the InternalData."));
}

template<int dim, typename number>
PatchWorker<dim, number>::PatchWorker(PatchInfo<dim> & patch_info_in)
  : patch_info(&patch_info_in),
    patch_size(UniversalInfo<dim>::n_cells(patch_info_in.get_additional_data().patch_variant))
{
  AssertThrow(patch_info_in.get_additional_data().patch_variant != TPSS::PatchVariant::invalid,
              ExcInvalidState());

  /**
   * If the given patch_info_in is already initialized and in valid state there is no work
   */
  typename PatchInfo<dim>::PartitionData subdomain_partition_data;
  compute_partition_data(subdomain_partition_data, patch_info_in.get_internal_data());
  if(subdomain_partition_data.check_compatibility(patch_info_in.subdomain_partition_data))
    return;

  /**
   * Initialize the vectorized infrastructure for the patches stored
   * in the PatchInfo. Partitioning into boundary/interior and
   * complete/incomplete patches is presented by @p
   * PatchInfo::subdomain_partition_data and the associated strides
   * through with respect to the flat data array are set in
   * PatchInfo::patch_starts
   */
  clear_patch_info(patch_info_in);
  partition_patches(patch_info_in);
  const auto & partition_data = patch_info_in.subdomain_partition_data;
  Assert(partition_data.n_colors() > 0, ExcMessage("At least one color."));
  for(unsigned color = 0; color < partition_data.n_colors(); ++color)
    has_valid_state(partition_data, color);
}

template<int dim, typename number>
inline unsigned int
PatchWorker<dim, number>::n_physical_subdomains() const
{
  Assert(patch_info != nullptr, ExcNotInitialized());
  const auto n_physical_subdomains_per_partition = [this](const unsigned int partition,
                                                          const unsigned int color) {
    const auto & partition_data = patch_info->subdomain_partition_data;
    const auto & range          = partition_data.get_patch_range(partition, color);

    // TODO
    unsigned int n_subdomains = 0;
    for(unsigned id = range.first; id < range.second; ++id)
      n_subdomains += n_lanes_filled(id);
    return n_subdomains;
    // return (range.second - range.first) * macro_size;
  };

  unsigned     n_subdomains   = 0;
  const auto & partition_data = patch_info->subdomain_partition_data;
  const auto   n_colors       = partition_data.n_colors();
  for(unsigned color = 0; color < n_colors; ++color)
  {
    const auto n_partitions = partition_data.n_partitions(color);
    for(unsigned partition = 0; partition < n_partitions; ++partition)
      n_subdomains += n_physical_subdomains_per_partition(partition, color);
  }

  return n_subdomains;
}

template<int dim, typename number>
inline unsigned int
PatchWorker<dim, number>::n_lanes_filled(const unsigned int patch_id) const
{
  Assert(patch_info != nullptr, ExcNotInitialized());
  AssertIndexRange(patch_id, patch_info->subdomain_partition_data.n_subdomains());

  return patch_info->n_lanes_filled[patch_id];
}

template<int dim, typename number>
inline const typename PatchInfo<dim>::PartitionData &
PatchWorker<dim, number>::get_partition_data() const
{
  Assert(patch_info != nullptr, ExcNotInitialized());
  return patch_info->subdomain_partition_data;
}

template<int dim, typename number>
inline std::vector<
  std::array<typename PatchWorker<dim, number>::CellIterator, PatchWorker<dim, number>::macro_size>>
PatchWorker<dim, number>::get_cell_collection(unsigned int patch) const
{
  using namespace dealii;
  Assert(patch_info != nullptr, ExcNotInitialized());
  AssertIndexRange(patch, patch_info->subdomain_partition_data.n_subdomains());

  std::vector<std::array<CellIterator, macro_size>> cell_collect(patch_size);
  const auto & cell_iterators = patch_info->get_internal_data()->cell_iterators;
  const auto & patch_starts   = patch_info->patch_starts;
  auto         cell_it        = cell_iterators.cbegin() + patch_starts[patch];
  if(n_lanes_filled(patch) < macro_size) // incomplete
  {
    Assert(n_lanes_filled(patch) == 1, ExcMessage("TODO"));
    for(unsigned int cell_no = 0; cell_no < patch_size; ++cell_it, ++cell_no)
      std::fill(cell_collect[cell_no].begin(), cell_collect[cell_no].end(), *cell_it);
  }
  else // complete
  {
    for(unsigned int m = 0; m < macro_size; ++m)
      for(unsigned int cell_no = 0; cell_no < patch_size; ++cell_it, ++cell_no)
        cell_collect[cell_no][m] = *cell_it;
  }

  return cell_collect;
}

template<int dim, typename number>
inline std::vector<ArrayView<const typename PatchWorker<dim, number>::CellIterator>>
PatchWorker<dim, number>::get_cell_collection_views(unsigned int patch_id) const
{
  Assert(patch_info != nullptr, ExcNotInitialized());
  AssertIndexRange(patch_id, patch_info->subdomain_partition_data.n_subdomains());

  std::vector<ArrayView<const CellIterator>> views;
  const auto & cell_iterators = patch_info->get_internal_data()->cell_iterators;
  const auto & patch_starts   = patch_info->patch_starts;
  const auto   begin          = cell_iterators.data() + patch_starts[patch_id];

  for(unsigned int m = 0; m < n_lanes_filled(patch_id); ++m)
  {
    const auto first = begin + m * patch_size;
    views.emplace_back(ArrayView<const CellIterator>(first, patch_size));
  }

  return views;
}

template<int dim, typename number>
inline void
PatchWorker<dim, number>::compute_partition_data(
  typename PatchInfo<dim>::PartitionData &      partition_data,
  const typename PatchInfo<dim>::InternalData * internal_data)
{
  AssertDimension(internal_data->n_interior_subdomains.size(),
                  internal_data->n_boundary_subdomains.size());
  const unsigned int n_colors = internal_data->n_interior_subdomains.size();
  partition_data.clear();
  partition_data.partitions.resize(n_colors);
  unsigned int n_subdomains_before = 0;
  for(unsigned int color = 0; color < n_colors; ++color)
  {
    auto & partitions = partition_data.partitions[color];
    partitions.resize(5);
    const unsigned int n_interior_subdomains = (internal_data->n_interior_subdomains)[color];
    const unsigned int n_boundary_subdomains = (internal_data->n_boundary_subdomains)[color];

    partitions[0] = n_subdomains_before;
    partitions[1] = partitions[0] + (n_interior_subdomains % macro_size); // interior incomplete
    partitions[2] = partitions[1] + (n_boundary_subdomains % macro_size); // boundary incomplete
    partitions[3] = partitions[2] + (n_interior_subdomains / macro_size); // interior complete
    partitions[4] = partitions[3] + (n_boundary_subdomains / macro_size); // boundary complete
    n_subdomains_before = partitions.back();

    has_valid_state(partition_data, color);
  }
}

template<int dim, typename number>
inline void
PatchWorker<dim, number>::has_valid_state(
  const typename PatchInfo<dim>::PartitionData & subdomain_partition_data,
  const unsigned int                             color)
{
  const auto & partitions = subdomain_partition_data.partitions;
  AssertDimension(partitions[color].size(), 5);
  for(unsigned int pid = 0; pid < partitions[color].size() - 1; ++pid)
    Assert(partitions[color][pid] >= 0 && partitions[color][pid] <= partitions[color][pid + 1],
           dealii::ExcInvalidState());
}

} // end namespace TPSS

#include "patch_info.templates.h"

#endif // end inclusion guard
