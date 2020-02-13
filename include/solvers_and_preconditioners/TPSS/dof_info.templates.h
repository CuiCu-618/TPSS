namespace TPSS
{
template<int dim, typename Number>
void
DoFInfo<dim, Number>::initialize(
  const DoFHandler<dim> *                                                   dof_handler_in,
  const PatchInfo<dim> *                                                    patch_info_in,
  const internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> * shape_info_in,
  const AdditionalData &                                                    additional_data_in)

{
  Assert(patch_info_in->get_internal_data()->level != numbers::invalid_unsigned_int,
         ExcMessage("Handles level cells only."));
  Assert(DoFLayout::invalid != TPSS::get_dof_layout(dof_handler_in->get_fe()),
         ExcMessage("Finite element is not supported."));
  Assert(additional_data_in.level != numbers::invalid_unsigned_int,
         ExcMessage("Implemented for level cells only"));

  clear();
  dof_handler = dof_handler_in;
  patch_info  = patch_info_in;
  shape_infos.fill(shape_info_in);
  additional_data = additional_data_in;

  initialize_impl();
}


template<int dim, typename Number>
void
DoFInfo<dim, Number>::initialize(
  const DoFHandler<dim> * dof_handler_in,
  const PatchInfo<dim> *  patch_info_in,
  const std::array<internal::MatrixFreeFunctions::ShapeInfo<VectorizedArray<Number>> *, dim> &
                         shape_infos_in,
  const AdditionalData & additional_data_in)
{
  Assert(patch_info_in->get_internal_data()->level != numbers::invalid_unsigned_int,
         ExcMessage("Handles level cells only."));
  Assert(DoFLayout::invalid != TPSS::get_dof_layout(dof_handler_in->get_fe()),
         ExcMessage("Finite element is not supported."));
  Assert(additional_data_in.level != numbers::invalid_unsigned_int,
         ExcMessage("Implemented for level cells only"));

  clear();
  dof_handler = dof_handler_in;
  patch_info  = patch_info_in;
  std::copy(shape_infos_in.cbegin(), shape_infos_in.cend(), shape_infos.begin());
  additional_data = additional_data_in;

  initialize_impl();
}

template<int dim, typename Number>
void
DoFInfo<dim, Number>::initialize_impl()
{
  /// fill the lexicographic-to-hierarchic-numbering map
  if(DoFLayout::Q == get_dof_layout())
    l2h = FETools::lexicographic_to_hierarchic_numbering(dof_handler->get_fe());

  /// store global dof indices
  const auto n_cells = patch_info->n_cells_plain();
  dof_indices.reserve(n_cells);
  for(auto i = 0U; i < n_cells; ++i)
    dof_indices.emplace_back(get_level_dof_indices(i));
  dof_indices.shrink_to_fit();
}


template<int dim, typename number>
std::shared_ptr<const Utilities::MPI::Partitioner>
PatchDoFWorker<dim, number>::initialize_vector_partitioner() const
{
  const auto     level         = dof_info->get_additional_data().level;
  const IndexSet owned_indices = std::move(dof_info->dof_handler->locally_owned_mg_dofs(level));

  // Note: For certain meshes (in particular in 3D and with many
  // processors), it is really necessary to cache intermediate data. After
  // trying several objects such as std::set, a vector that is always kept
  // sorted, and a vector that is initially unsorted and sorted once at the
  // end, the latter has been identified to provide the best performance.
  // Martin Kronbichler
  const auto   my_subdomain_id   = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto & is_ghost_on_level = [my_subdomain_id](const auto & cell) {
    const bool is_owned      = cell->level_subdomain_id() == my_subdomain_id;
    const bool is_artificial = cell->level_subdomain_id() == numbers::artificial_subdomain_id;
    return !is_owned && !is_artificial;
  };
  std::vector<types::global_dof_index> dof_indices;
  std::vector<types::global_dof_index> dofs_on_ghosts;
  const unsigned int n_colors = patch_worker_type::get_partition_data().n_colors();
  for(unsigned int color = 0; color < n_colors; ++color)
  {
    const auto range = patch_worker_type::get_ghost_range(color);
    for(unsigned int patch_id = range.first; patch_id < range.second; ++patch_id)
    {
      const auto & collection_views = patch_worker_type::get_cell_collection_views(patch_id);
      for(const auto & collection : collection_views)
        for(const auto & cell : collection)
          if(is_ghost_on_level(cell))
          {
            dof_indices.resize(cell->get_fe().dofs_per_cell);
            cell->get_mg_dof_indices(dof_indices);
            for(const auto dof_index : dof_indices)
              if(!owned_indices.is_element(dof_index))
                dofs_on_ghosts.push_back(dof_index);
          }
    }
  }

  /// sort, compress out duplicates, fill into index set
  std::sort(dofs_on_ghosts.begin(), dofs_on_ghosts.end());
  IndexSet   ghost_indices(owned_indices.size());
  const auto end_without_duplicates = std::unique(dofs_on_ghosts.begin(), dofs_on_ghosts.end());
  ghost_indices.add_indices(dofs_on_ghosts.begin(), end_without_duplicates);
  ghost_indices.compress();
  IndexSet larger_ghost_indices(owned_indices.size());
  DoFTools::extract_locally_relevant_level_dofs(*(dof_info->dof_handler),
                                                level,
                                                larger_ghost_indices);
  larger_ghost_indices.subtract_set(owned_indices);

  const auto partitioner =
    std::make_shared<Utilities::MPI::Partitioner>(owned_indices, MPI_COMM_WORLD);
  partitioner->set_ghost_indices(ghost_indices, larger_ghost_indices);
  // TODO ?
  // partitioner->set_ghost_indices(larger_ghost_indices);

  // // DEBUG
  // std::ostringstream oss;
  // oss << "info on mpi proc " << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << " on level
  // " << level << std::endl; oss << "ghost index set:" << std::endl; for (const auto i :
  // ghost_indices)
  //   oss << i << " ";
  // oss << std::endl;
  // oss << "larger ghost index set:" << std::endl;
  // for (const auto i : larger_ghost_indices)
  //   oss << i << " ";
  // oss << std::endl;
  // const auto import_targets = partitioner->import_targets();
  // oss << "import targets: (proc, n_targets)" << std::endl;
  // for (const auto p : import_targets)
  //   oss << "(" << p.first << ", " << p.second << ") ";
  // oss << std::endl;
  // const auto ghost_targets = partitioner->ghost_targets();
  // oss << "ghost targets: (proc, n_targets)" << std::endl;
  // for (const auto p : ghost_targets)
  //   oss << "(" << p.first << ", " << p.second << ") ";
  // oss << std::endl;
  // std::cout << oss.str() << std::endl;

  return partitioner;
}


} // end namespace TPSS
