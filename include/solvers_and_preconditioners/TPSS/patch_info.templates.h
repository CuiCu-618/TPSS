namespace TPSS
{
template<int dim>
void
PatchInfo<dim>::initialize(const DoFHandler<dim> * dof_handler,
                           const AdditionalData    additional_data_in)
{
  clear();

  Assert(!(dof_handler->get_triangulation().has_hanging_nodes()),
         ExcMessage("Not implemented for adaptive meshes!"));
  Assert(additional_data_in.level != static_cast<unsigned int>(-1),
         ExcMessage("Implemented for level cell iterators!"));
  AssertIndexRange(additional_data_in.level, dof_handler->get_triangulation().n_global_levels());

  // *** submit additional data
  additional_data           = additional_data_in;
  internal_data.level       = additional_data.level;
  internal_data.dof_handler = dof_handler;

  // *** extract and colorize subdomains depending on the patch variant
  if(additional_data.patch_variant == TPSS::PatchVariant::cell)
    initialize_cell_patches(dof_handler, additional_data);
  else if(additional_data.patch_variant == TPSS::PatchVariant::vertex)
    initialize_vertex_patches(dof_handler, additional_data);
  else
    AssertThrow(false, dealii::ExcNotImplemented());
  const auto n_cells_stored_after_init = internal_data.cell_iterators.size();
  (void)n_cells_stored_after_init;

  // *** store internal data
  internal_data.triangulation = &(dof_handler->get_triangulation());
  // TODO we should not need to store the dof handler
  internal_data.dof_handler = dof_handler;

  internal_data.cell_iterators.shrink_to_fit();
  internal_data.cell_level_and_index_pairs.clear();
  std::transform(internal_data.cell_iterators.cbegin(),
                 internal_data.cell_iterators.cend(),
                 std::back_inserter(internal_data.cell_level_and_index_pairs),
                 [](const auto & cell) {
                   return std::make_pair<int, int>(cell->level(), cell->index());
                 });
  internal_data.cell_iterators.clear();
  iterator_is_cached        = !internal_data.cell_iterators.empty();
  level_and_index_is_cached = !internal_data.cell_level_and_index_pairs.empty();

  /// check validity
  AssertDimension(n_cells_stored_after_init, n_cells_plain());
  const auto n_colors_mpimin = Utilities::MPI::min(internal_data.n_colors(), MPI_COMM_WORLD);
  const auto n_colors_mpimax = Utilities::MPI::max(internal_data.n_colors(), MPI_COMM_WORLD);
  (void)n_colors_mpimin, (void)n_colors_mpimax;
  Assert(n_colors_mpimin == n_colors_mpimax,
         ExcMessage("No unified number of colors between mpi-procs."));
  Assert(!internal_data.empty_on_all(), ExcMessage("No mpi-proc owns a patch!"));
}


template<int dim>
void
PatchInfo<dim>::initialize_cell_patches(const dealii::DoFHandler<dim> * dof_handler,
                                        const AdditionalData            additional_data)
{
  const auto level        = additional_data.level;
  const auto color_scheme = additional_data.smoother_variant;

  Timer time;
  time.restart();

  /**
   * Gathering the locally owned cell iterators as collection of cells
   * (patch). Here, it is only one cell iterator per collection.
   */
  const auto locally_owned_range_mg =
    filter_iterators(dof_handler->mg_cell_iterators_on_level(level),
                     IteratorFilters::LocallyOwnedLevelCell());
  std::vector<std::vector<CellIterator>> cell_collections;
  for(const auto & cell : locally_owned_range_mg)
  {
    std::vector<CellIterator> patch;
    patch.push_back(cell);
    cell_collections.emplace_back(patch);
  }

  time.stop();
  time_data.emplace_back(time.wall_time(), "Cell-based gathering");
  time.restart();

  /**
   * Coloring of the "cell patches". For the additive operator, we only have one
   * color. However, we require a vector of PatchIterators to call
   * submit_patches.
   */
  std::vector<std::vector<PatchIterator>> colored_iterators;
  constexpr int regular_size = UniversalInfo<dim>::n_cells(PatchVariant::cell);
  if(color_scheme == TPSS::SmootherVariant::additive) // ADDITIVE
  {
    colored_iterators.resize(1);
    std::vector<PatchIterator> & patch_iterators = colored_iterators.front();
    for(auto patch = cell_collections.cbegin(); patch != cell_collections.cend(); ++patch)
      patch_iterators.emplace_back(patch);
  }

  /**
   * Coloring of the "cell patches". For the multiplicative algorithm,
   * one has to prevent that two local solvers sharing a (global)
   * degree of freedom are applied to the same residual vector. For
   * example, DG elements are coupled in terms of the face integrals
   * involving traces of both elements. Therefore, we cells are in
   * conflict if they share a common face. TODO other FE types!
   */
  else if(color_scheme == TPSS::SmootherVariant::multiplicative) // MULTIPLICATIVE
  {
    const bool do_graph_coloring = !additional_data.coloring_func;
    if(do_graph_coloring) // graph coloring
    {
      const bool is_mpi_parallel = (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1);
      AssertThrow(!is_mpi_parallel,
                  ExcMessage("Graph coloring is not compatible with distributed triangulations."));
      colored_iterators = std::move(GraphColoring::make_graph_coloring(cell_collections.cbegin(),
                                                                       cell_collections.cend(),
                                                                       get_face_conflicts));
    }

    else // user-defined coloring
    {
      colored_iterators =
        std::move(additional_data.coloring_func(cell_collections, additional_data));
    }
  }

  else // color_scheme
    AssertThrow(false, ExcNotImplemented());

  time.stop();
  time_data.emplace_back(time.wall_time(), "Cell-based coloring");
  time.restart();

  const unsigned int n_colors = colored_iterators.size();
  for(unsigned int color = 0; color < n_colors; ++color)
    submit_patches<regular_size>(colored_iterators[color]);
  count_physical_subdomains();

  time.stop();
  time_data.emplace_back(time.wall_time(), "Submit cell-based patches");
  time.restart();

  // *** check if the InternalData is valid
  if(color_scheme == TPSS::SmootherVariant::additive)
    AssertDimension(internal_data.n_colors(), 1);
  const unsigned int n_physical_subdomains =
    internal_data.subdomain_quantities_accumulated.n_interior +
    internal_data.subdomain_quantities_accumulated.n_boundary;
  (void)n_physical_subdomains;
  AssertDimension(n_physical_subdomains, internal_data.cell_iterators.size());

  if(additional_data.visualize_coloring)
    additional_data.visualize_coloring(*dof_handler, colored_iterators, "cp_");

  // *** print detailed information
  if(additional_data.print_details)
  {
    print_row_variable(pcout, 45, "Coloring on level:", additional_data.level);
    print_row_variable(
      pcout, 5, "", 10, "color:", 30, "# of interior patches:", 30, "# of boundary patches:");
    const auto n_colors       = internal_data.subdomain_quantities.size();
    auto       subdomain_data = internal_data.subdomain_quantities.cbegin();
    for(unsigned c = 0; c < n_colors; ++c, ++subdomain_data)
      print_row_variable(
        pcout, 5, "", 10, c, 30, subdomain_data->n_interior, 30, subdomain_data->n_boundary);
    pcout << std::endl;
  }
}


template<int dim>
std::vector<std::vector<typename PatchInfo<dim>::CellIterator>>
PatchInfo<dim>::gather_vertex_patches(const DoFHandler<dim> & dof_handler,
                                      const AdditionalData &  additional_data) const
{
  const unsigned int level = additional_data.level;

  // LAMBDA checks if a vertex is at the physical boundary
  auto && is_boundary_vertex = [](const CellIterator & cell, const unsigned int vertex_id) {
    return std::any_of(std::begin(GeometryInfo<dim>::vertex_to_face[vertex_id]),
                       std::end(GeometryInfo<dim>::vertex_to_face[vertex_id]),
                       [&cell](const auto & face_no) { return cell->at_boundary(face_no); });
  };
  constexpr unsigned int regular_vpatch_size = 1 << dim;
  const auto &           tria                = dof_handler.get_triangulation();
  const auto             locally_owned_range_mg =
    filter_iterators(dof_handler.mg_cell_iterators_on_level(level),
                     IteratorFilters::LocallyOwnedLevelCell());

  /**
   * A mapping @p global_to_local_map between the global vertex and
   * the pair containing the number of locally owned cells and the
   * number of all cells (including ghosts) is constructed
   */
  std::map<unsigned int, std::pair<unsigned int, unsigned int>> global_to_local_map;
  for(const auto & cell : locally_owned_range_mg)
  {
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
      if(!is_boundary_vertex(cell, v))
      {
        const unsigned int global_index = cell->vertex_index(v);
        const auto         element      = global_to_local_map.find(global_index);
        if(element != global_to_local_map.cend())
        {
          ++(element->second.first);
          ++(element->second.second);
        }
        else
        {
          const auto n_cells_pair = std::pair<unsigned, unsigned>{1, 1};
          const auto status =
            global_to_local_map.insert(std::make_pair(global_index, n_cells_pair));
          (void)status;
          Assert(status.second, ExcMessage("failed to insert key-value-pair"))
        }
      }
  }

  /**
   * Ghost patches are stored as the mapping @p global_to_ghost_id
   * between the global vertex index and GhostPatch. The number of
   * cells, book-kept in @p global_to_local_map, is updated taking the
   * ghost cells into account.
   */
  // TODO: is_ghost_on_level() missing
  const auto not_locally_owned_range_mg =
    filter_iterators(dof_handler.mg_cell_iterators_on_level(level),
                     [](const auto & cell) { return !(cell->is_locally_owned_on_level()); });
  std::map<unsigned int, GhostPatch> global_to_ghost_id;
  for(const auto & cell : not_locally_owned_range_mg)
  {
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      const unsigned int global_index = cell->vertex_index(v);
      const auto         element      = global_to_local_map.find(global_index);
      if(element != global_to_local_map.cend())
      {
        ++(element->second.second);
        const unsigned int subdomain_id_ghost = cell->level_subdomain_id();
        const auto         ghost              = global_to_ghost_id.find(global_index);
        if(ghost != global_to_ghost_id.cend())
          ghost->second.submit_id(subdomain_id_ghost, cell->id());
        else
        {
          const auto status =
            global_to_ghost_id.emplace(global_index, GhostPatch(subdomain_id_ghost, cell->id()));
          (void)status;
          Assert(status.second, ExcMessage("failed to insert key-value-pair"));
        }
      }
    }
  }

  { // ASSIGN GHOSTS
    const unsigned int my_subdomain_id = tria.locally_owned_subdomain();
    /**
     * logic: if the mpi-proc owns more than half of the cells within
     *        a ghost patch he takes ownership
     */
    {
      //: (1) add subdomain_ids of locally owned cells to GhostPatches
      for(const auto & cell : locally_owned_range_mg)
        for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          const unsigned global_index = cell->vertex_index(v);
          const auto     ghost        = global_to_ghost_id.find(global_index);
          //: checks if the global vertex has ghost cells attached
          if(ghost != global_to_ghost_id.end())
            ghost->second.submit_id(my_subdomain_id, cell->id());
        }

      std::set<unsigned> to_be_owned;
      std::set<unsigned> to_be_erased;
      for(const auto &key_value : global_to_ghost_id)
      {
        const unsigned int global_index     = key_value.first;
        const auto &       proc_to_cell_ids = key_value.second.proc_to_cell_ids;

        const auto & get_proc_with_most_cellids = [](const auto & lhs, const auto & rhs) {
          const std::vector<CellId> & cell_ids_lhs = lhs.second;
          const std::vector<CellId> & cell_ids_rhs = rhs.second;
          Assert(!cell_ids_lhs.empty(), ExcMessage("should not be empty"));
          Assert(!cell_ids_rhs.empty(), ExcMessage("should not be empty"));
          return (cell_ids_lhs.size() < cell_ids_rhs.size());
        };

        const auto         most                       = std::max_element(proc_to_cell_ids.cbegin(),
                                           proc_to_cell_ids.cend(),
                                           get_proc_with_most_cellids);
        const unsigned int subdomain_id_most          = most->first;
        const unsigned int n_locally_owned_cells_most = most->second.size();
        const auto         member                     = global_to_local_map.find(global_index);
        Assert(member != global_to_local_map.cend(), ExcMessage("must be listed as patch"));
        const unsigned int n_cells = member->second.second;
        if(my_subdomain_id == subdomain_id_most)
        {
          AssertDimension(member->second.first, n_locally_owned_cells_most);
          if(2 * n_locally_owned_cells_most > n_cells)
            to_be_owned.insert(global_index);
        }
        else
        {
          if(2 * n_locally_owned_cells_most > n_cells)
            to_be_erased.insert(global_index);
        }
      }

      for(const unsigned global_index : to_be_owned)
      {
        auto & my_patch = global_to_local_map[global_index];
        my_patch.first  = my_patch.second;
        global_to_ghost_id.erase(global_index);
      }
      for(const unsigned global_index : to_be_erased)
      {
        global_to_local_map.erase(global_index);
        global_to_ghost_id.erase(global_index);
      }
    }

    /**
     * logic: the owner of the cell with the lowest CellId takes ownership
     */
    {
      //: (2) determine mpi-proc with the minimal CellId for all GhostPatches
      std::set<unsigned> to_be_owned;
      for(const auto &key_value : global_to_ghost_id)
      {
        const unsigned int global_index     = key_value.first;
        const auto &       proc_to_cell_ids = key_value.second.proc_to_cell_ids;

        const auto & get_proc_with_min_cellid = [](const auto & lhs, const auto & rhs) {
          std::vector<CellId> cell_ids_lhs = lhs.second;
          Assert(!cell_ids_lhs.empty(), ExcMessage("should not be empty"));
          std::sort(cell_ids_lhs.begin(), cell_ids_lhs.end());
          const auto          min_cell_id_lhs = cell_ids_lhs.front();
          std::vector<CellId> cell_ids_rhs    = rhs.second;
          Assert(!cell_ids_rhs.empty(), ExcMessage("should not be empty"));
          std::sort(cell_ids_rhs.begin(), cell_ids_rhs.end());
          const auto min_cell_id_rhs = cell_ids_rhs.front();
          return min_cell_id_lhs < min_cell_id_rhs;
        };

        const auto min = std::min_element(proc_to_cell_ids.cbegin(),
                                          proc_to_cell_ids.cend(),
                                          get_proc_with_min_cellid);

        const unsigned int subdomain_id_min = min->first;
        if(my_subdomain_id == subdomain_id_min)
          to_be_owned.insert(global_index);
      }

      //: (3) set owned GhostPatches in global_to_local_map and delete all remaining
      for(const unsigned global_index : to_be_owned)
      {
        auto & my_patch = global_to_local_map[global_index];
        my_patch.first  = my_patch.second;
        global_to_ghost_id.erase(global_index);
      }
      for(const auto &key_value : global_to_ghost_id)
      {
        const unsigned int global_index = key_value.first;
        global_to_local_map.erase(global_index);
      }
    }
  }

  /**
   * Enumerate the patches contained in @p global_to_local_map by
   * replacing the former number of locally owned cells in terms of a
   * consecutive numbering. The local numbering is required for
   * gathering the level cell iterators into a collection @
   * cell_collections according to the global vertex index.
   */
  unsigned int local_index = 0;
  for(auto & key_value : global_to_local_map)
  {
    key_value.second.first = local_index++;
  }
  const unsigned n_subdomains = global_to_local_map.size();
  AssertDimension(n_subdomains, local_index);
  std::vector<std::vector<CellIterator>> cell_collections;
  cell_collections.resize(n_subdomains);
  for(auto & cell : dof_handler.mg_cell_iterators_on_level(level))
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      const unsigned int global_index = cell->vertex_index(v);
      const auto         element      = global_to_local_map.find(global_index);
      if(element != global_to_local_map.cend())
      {
        const unsigned int local_index = element->second.first;
        const unsigned int patch_size  = element->second.second;
        auto &             collection  = cell_collections[local_index];
        if(collection.empty())
          collection.resize(patch_size);
        if(patch_size == regular_vpatch_size) // regular patch
          collection[regular_vpatch_size - 1 - v] = cell;
        else // irregular patch
          AssertThrow(false, ExcMessage("TODO irregular vertex patches"));
      }
    }

  return cell_collections;
}


template<int dim>
void
PatchInfo<dim>::initialize_vertex_patches(const dealii::DoFHandler<dim> * dof_handler,
                                          const AdditionalData            additional_data)
{
  constexpr auto regular_vpatch_size = UniversalInfo<dim>::n_cells(PatchVariant::vertex);
  const auto     color_scheme        = additional_data.smoother_variant;

  Timer time;
  time.restart();

  /**
   * Collecting the cell iterators attached to a vertex. See @p
   * gather_vertex_patches for more information.
   */
  std::vector<std::vector<CellIterator>> cell_collections;
  if(!additional_data.patch_distribution_func)
    cell_collections = std::move(gather_vertex_patches(*dof_handler, additional_data));
  else
    additional_data.patch_distribution_func(dof_handler, additional_data, cell_collections);

  time.stop();
  time_data.emplace_back(time.wall_time(), "Vertex patch gathering");
  time.restart();

  /**
   * Coloring of vertex patches. For the additive operator we only require one
   * color as long as we do not use thread parallelism. In multi-threaded loops
   * race-conditions might occur due to overlap. Overlap means that two adjacent
   * FE subspaces share a common global degree of freedom. Consequently, two
   * local solvers might simultaneously write to the same DoF entry in the
   * destination vector. At the moment the user has to provide a coloring scheme
   * that prevents race conditions!
   */
  std::string                             str_coloring_algorithm = "TBA";
  std::vector<std::vector<PatchIterator>> colored_iterators;
  switch(color_scheme)
  {
    case TPSS::SmootherVariant::additive:
    {
      if(!additional_data.use_tbb)
      {
        str_coloring_algorithm = "none";
        colored_iterators.resize(1);
        auto & patch_iterators = colored_iterators.front();
        for(auto it = cell_collections.cbegin(); it != cell_collections.cend(); ++it)
          patch_iterators.emplace_back(it);
      }

      else
      {
        AssertThrow(
          additional_data.coloring_func,
          ExcMessage(
            "The user is responsible to provide a coloring scheme avoiding race conditions in case of additive vertex patches."));
        str_coloring_algorithm = "user_avoidrace";
        colored_iterators =
          std::move(additional_data.coloring_func(cell_collections, additional_data));
      }

      break;
    }
    case TPSS::SmootherVariant::multiplicative:
    {
      if(!additional_data.coloring_func) // graph coloring
      {
        const bool is_mpi_parallel = (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1);
        AssertThrow(!is_mpi_parallel,
                    ExcMessage(
                      "Graph coloring is not compatible with distributed triangulations."));
        str_coloring_algorithm = "graph";
        colored_iterators = std::move(GraphColoring::make_graph_coloring(cell_collections.cbegin(),
                                                                         cell_collections.cend(),
                                                                         get_face_conflicts));
      }

      else // user-defined coloring
      {
        str_coloring_algorithm = "user";
        colored_iterators =
          std::move(additional_data.coloring_func(cell_collections, additional_data));
      }

      break;
    }
    default:
    {
      AssertThrow(false, ExcNotImplemented());
      break;
    }
  } // end switch

  std::ostringstream oss;
  oss << "Vertex patch coloring (" << str_coloring_algorithm << ")";
  time.stop();
  time_data.emplace_back(time.wall_time(), oss.str());
  time.restart();

  if(additional_data.visualize_coloring)
    additional_data.visualize_coloring(*dof_handler, colored_iterators, "vp_");

  /**
   * Submisson of the colored collections of CellIterators into the
   * InternalData.
   */
  const unsigned int n_colors = colored_iterators.size();
  for(unsigned int cc = 0; cc < n_colors; ++cc)
  {
    const unsigned int color = cc;
    submit_patches<regular_vpatch_size>(colored_iterators[color]);
  }
  count_physical_subdomains();

  time.stop();
  time_data.emplace_back(time.wall_time(), "Submit vertex patches");
  time.restart();

  // *** check if the InternalData is valid
  AssertDimension(internal_data.cell_iterators.size() % regular_vpatch_size, 0);
  // if(color_scheme == TPSS::SmootherVariant::additive)
  //   // TODO more colors to avoid race conditions ?
  //   AssertDimension(internal_data.n_colors(), 1);
  const unsigned int n_physical_subdomains =
    internal_data.subdomain_quantities_accumulated.n_interior +
    internal_data.subdomain_quantities_accumulated.n_boundary;
  (void)n_physical_subdomains;
  AssertDimension(n_physical_subdomains, internal_data.cell_iterators.size() / regular_vpatch_size);

  if(additional_data.print_details && color_scheme != TPSS::SmootherVariant::additive)
  {
    print_row_variable(pcout, 2, "", 43, oss.str(), additional_data.level);
    pcout << std::endl;

    print_row_variable(
      pcout, 5, "", 10, "color:", 30, "# of interior patches:", 30, "# of boundary patches:");
    const auto n_colors       = internal_data.subdomain_quantities.size();
    auto       subdomain_data = internal_data.subdomain_quantities.cbegin();
    for(unsigned c = 0; c < n_colors; ++c, ++subdomain_data)
      print_row_variable(
        pcout, 5, "", 10, c, 30, subdomain_data->n_interior, 30, subdomain_data->n_boundary);
    pcout << std::endl;
  }
}


template<int dim>
void
PatchInfo<dim>::count_physical_subdomains()
{
  internal_data.subdomain_quantities_accumulated.n_interior =
    std::accumulate(internal_data.subdomain_quantities.cbegin(),
                    internal_data.subdomain_quantities.cend(),
                    0,
                    [](const auto sum, const auto & data) { return sum + data.n_interior; });
  internal_data.subdomain_quantities_accumulated.n_boundary =
    std::accumulate(internal_data.subdomain_quantities.cbegin(),
                    internal_data.subdomain_quantities.cend(),
                    0,
                    [](const auto sum, const auto & data) { return sum + data.n_boundary; });
  internal_data.subdomain_quantities_accumulated.n_interior_ghost =
    std::accumulate(internal_data.subdomain_quantities.cbegin(),
                    internal_data.subdomain_quantities.cend(),
                    0,
                    [](const auto sum, const auto & data) { return sum + data.n_interior_ghost; });
  internal_data.subdomain_quantities_accumulated.n_boundary_ghost =
    std::accumulate(internal_data.subdomain_quantities.cbegin(),
                    internal_data.subdomain_quantities.cend(),
                    0,
                    [](const auto sum, const auto & data) { return sum + data.n_boundary_ghost; });
}


template<int dim>
std::vector<types::global_dof_index>
PatchInfo<dim>::get_face_conflicts(const PatchIterator & patch)
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


template<int dim>
template<int regular_size>
void
PatchInfo<dim>::submit_patches(const std::vector<PatchIterator> & patch_iterators)
{
  // First, submit all interior patches and subsequently all boundary patches
  const auto & internal_submit = [this](const std::vector<PatchIterator> & patch_iterators) {
    unsigned int               n_interior_subdomains_regular = 0;
    std::vector<PatchIterator> boundary_patch_regular;
    for(const auto & patch : patch_iterators)
    {
      const bool patch_at_boundary =
        std::any_of(patch->cbegin(), patch->cend(), IteratorFilters::AtBoundary{});
      if(patch->size() == regular_size) // regular
      {
        if(patch_at_boundary)
        {
          boundary_patch_regular.push_back(patch);
        }
        else
        {
          ++n_interior_subdomains_regular;
          for(const auto & cell : *patch)
            internal_data.cell_iterators.emplace_back(cell);
        }
      }
      else // irregular
        Assert(false, ExcNotImplemented());
    }

    for(const auto it : boundary_patch_regular)
    {
      for(const auto & cell : *it)
        internal_data.cell_iterators.emplace_back(cell);
    }

    SubdomainData local_data;
    local_data.n_interior = n_interior_subdomains_regular;
    local_data.n_boundary = boundary_patch_regular.size();
    return local_data;
  };
  // We separate the patch iterators into locally owned subdomains
  // and those with ghost cells. First, we submit locally owned and
  // subsequently subdomains with ghosts. Each group is separated
  // into interior (first) and boundary (second) subdomains,
  // respectively.
  const auto   my_subdomain_id   = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto & is_ghost_on_level = [my_subdomain_id](const auto & cell) {
    const bool is_owned      = cell->level_subdomain_id() == my_subdomain_id;
    const bool is_artificial = cell->level_subdomain_id() == numbers::artificial_subdomain_id;
    return !is_owned && !is_artificial;
  };
  std::vector<PatchIterator> owned_patch_iterators, ghost_patch_iterators;
  for(const auto & patch : patch_iterators)
  {
    const bool patch_is_ghost = std::any_of(patch->cbegin(), patch->cend(), is_ghost_on_level);
    if(patch_is_ghost)
      ghost_patch_iterators.emplace_back(patch);
    else
      owned_patch_iterators.emplace_back(patch);
  }
  AssertDimension(owned_patch_iterators.size() + ghost_patch_iterators.size(),
                  patch_iterators.size());
  const bool is_mpi_parallel = (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1);
  if(!is_mpi_parallel)
    AssertDimension(ghost_patch_iterators.size(), 0);

  const auto    owned_subdomain_data = internal_submit(owned_patch_iterators);
  const auto    ghost_subdomain_data = internal_submit(ghost_patch_iterators);
  SubdomainData subdomain_data;
  subdomain_data.n_interior = owned_subdomain_data.n_interior + ghost_subdomain_data.n_interior;
  subdomain_data.n_boundary = owned_subdomain_data.n_boundary + ghost_subdomain_data.n_boundary;
  subdomain_data.n_interior_ghost = ghost_subdomain_data.n_interior;
  subdomain_data.n_boundary_ghost = ghost_subdomain_data.n_boundary;
  internal_data.subdomain_quantities.emplace_back(subdomain_data);
}


// template<int dim>
// void
// PatchInfo<dim>::write_visual_data(
//   const dealii::DoFHandler<dim> &                            dof_handler,
//   const std::vector<std::pair<unsigned int, unsigned int>> & reordered_colors) const
// {
//   constexpr auto     regular_vpatch_size = UniversalInfo<dim>::n_cells(PatchVariant::vertex);
//   const auto &       tria                = dof_handler.get_triangulation();
//   const unsigned int level               = internal_data.level;

//   if(level == tria.n_levels() - 1)
//   {
//     GridOutFlags::Svg gridout_flags;
//     gridout_flags.coloring           = GridOutFlags::Svg::Coloring::material_id;
//     gridout_flags.label_level_number = false;
//     gridout_flags.label_cell_index   = false;
//     gridout_flags.label_material_id  = true;
//     GridOut gridout;
//     gridout.set_flags(gridout_flags);

//     // *** set all material ids to void
//     const unsigned int void_id = 999;
//     CellIterator       cell = dof_handler.begin_mg(level), end_cell = dof_handler.end_mg(level);
//     for(; cell != end_cell; ++cell)
//       cell->set_material_id(void_id);

//     auto               cell_it  = internal_data.cell_iterators.cbegin();
//     const unsigned int n_colors = reordered_colors.size();
//     for(unsigned int color = 0; color < n_colors; ++color)
//     {
//       std::string filename = "make_graph_coloring_";
//       filename             = filename + "L" + Utilities::int_to_string(level) + "_COLOR" +
//                  Utilities::int_to_string(color, 2);
//       std::ofstream output((filename + ".svg").c_str());

//       unsigned int n_colored_cells = reordered_colors[color].first * regular_vpatch_size;
//       for(unsigned int c = 0; c < n_colored_cells; ++cell_it, ++c)
//         (*cell_it)->set_material_id(color);
//       gridout.write_svg(tria, output);

//       // *** reset all material ids to void
//       CellIterator cell = dof_handler.begin_mg(level), end_cell = dof_handler.end_mg(level);
//       for(; cell != end_cell; ++cell)
//         cell->set_material_id(void_id);
//     }
//     Assert(internal_data.cell_iterators.cend() == cell_it, ExcInternalError());
//   }
// }



} // end namespace TPSS
