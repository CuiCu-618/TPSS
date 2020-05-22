namespace TPSS
{
template<int dim, typename Number, int fe_degree>
template<typename VectorType>
AlignedVector<VectorizedArray<Number>>
PatchTransfer<dim, Number, fe_degree>::gather(const VectorType & src) const
{
  AssertIndexRange(patch_id, subdomain_handler.get_partition_data().n_subdomains());
  AssertDimension(src.size(), subdomain_handler.get_dof_handler().n_dofs(level));

  AlignedVector<VectorizedArray<Number>> dst(n_dofs_per_patch());
  for(unsigned int lane = 0; lane < macro_size; ++lane)
  {
    const auto & global_dof_indices = get_dof_indices(lane);
    AssertDimension(dst.size(), global_dof_indices.size());
    auto dof_index = global_dof_indices.cbegin();
    for(auto dst_value = dst.begin(); dst_value != dst.end(); ++dof_index, ++dst_value)
      (*dst_value)[lane] = internal::local_element(src, *dof_index);
  }

  AssertDimension(dst.size(), n_dofs_per_patch());
  return dst;
}


template<int dim, typename Number, int fe_degree>
template<typename VectorType>
void
PatchTransfer<dim, Number, fe_degree>::gather_add(const ArrayView<VectorizedArray<Number>> dst,
                                                  const VectorType & src) const
{
  AssertDimension(dst.size(), n_dofs_per_patch());
  const auto & src_local = gather(src);
  std::transform(dst.begin(),
                 dst.end(),
                 src_local.begin(),
                 dst.begin(),
                 [](const auto & dst, const auto & src) { return dst + src; });
}


template<int dim, typename Number, int fe_degree>
template<typename VectorType>
void
PatchTransfer<dim, Number, fe_degree>::gather_add(AlignedVector<VectorizedArray<Number>> & dst,
                                                  const VectorType & src) const
{
  AssertDimension(dst.size(), n_dofs_per_patch());
  const auto dst_view = make_array_view<VectorizedArray<Number>>(dst.begin(), dst.end());
  gather_add(dst_view, src);
}


template<int dim, typename Number, int fe_degree>
template<typename VectorType>
void
PatchTransfer<dim, Number, fe_degree>::scatter_add(
  VectorType &                                   dst,
  const ArrayView<const VectorizedArray<Number>> src) const
{
  AssertIndexRange(patch_id, subdomain_handler.get_partition_data().n_subdomains());
  AssertDimension(dst.size(), subdomain_handler.get_dof_handler().n_dofs(level));
  AssertDimension(src.size(), n_dofs_per_patch());

  for(unsigned int lane = 0; lane < patch_dof_worker.n_lanes_filled(patch_id); ++lane)
  {
    const auto & global_dof_indices = get_dof_indices(lane);
    AssertDimension(src.size(), global_dof_indices.size());
    auto dof_index = global_dof_indices.cbegin();
    for(auto src_value = src.cbegin(); src_value != src.cend(); ++dof_index, ++src_value)
      internal::local_element(dst, *dof_index) += (*src_value)[lane];
  }
}


template<int dim, typename Number, int fe_degree>
template<typename VectorType>
void
PatchTransfer<dim, Number, fe_degree>::scatter_add(
  VectorType &                                   dst,
  const AlignedVector<VectorizedArray<Number>> & src) const
{
  const auto src_view = make_array_view<const VectorizedArray<Number>>(src.begin(), src.end());
  scatter_add(dst, src_view);
}



// -----------------------------   PatchTransferBlock   ----------------------------



template<int dim, typename Number, int fe_degree>
AlignedVector<VectorizedArray<Number>>

PatchTransferBlock<dim, Number, fe_degree>::gather(const BlockVectorType & src) const
{
  AlignedVector<VectorizedArray<Number>> dst;
  reinit_local_vector(dst);
  auto begin = dst.begin();
  for(std::size_t b = 0; b < n_blocks; ++b)
  {
    const auto                         transfer = transfers[b];
    const auto                         size     = transfer->n_dofs_per_patch();
    ArrayView<VectorizedArray<Number>> dst_block{begin, size};
    transfer->gather_add(dst_block, src.block(b));
    begin += size;
  }
  AssertThrow(begin == dst.end(), ExcMessage("Inconsistent slicing."));
  return dst;
}


template<int dim, typename Number, int fe_degree>
void
PatchTransferBlock<dim, Number, fe_degree>::gather_add(AlignedVector<VectorizedArray<Number>> & dst,
                                                       const BlockVectorType & src) const
{
  AssertDimension(dst.size(), n_dofs_per_patch());
  const auto & src_local = gather(src);
  std::transform(dst.begin(),
                 dst.end(),
                 src_local.begin(),
                 dst.begin(),
                 [](const auto & dst, const auto & src) { return dst + src; });
}


template<int dim, typename Number, int fe_degree>
void
PatchTransferBlock<dim, Number, fe_degree>::scatter_add(
  BlockVectorType &                              dst,
  const AlignedVector<VectorizedArray<Number>> & src) const
{
  auto begin = src.begin();
  for(std::size_t b = 0; b < n_blocks; ++b)
  {
    const auto                               transfer = transfers[b];
    const auto                               size     = transfer->n_dofs_per_patch();
    ArrayView<const VectorizedArray<Number>> src_block{begin, size};
    transfer->scatter_add(dst.block(b), src_block);
    begin += size;
  }
  AssertThrow(begin == src.end(), ExcMessage("Inconsistent slicing."));
}



} // end namespace TPSS
