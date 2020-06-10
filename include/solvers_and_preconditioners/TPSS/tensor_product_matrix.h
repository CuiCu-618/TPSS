/*
 * tensor_product_matrix.h
 *
 *  Created on: Dec 04, 2019
 *      Author: witte
 */

#ifndef TENSOR_PRODUCT_MATRIX_H_
#define TENSOR_PRODUCT_MATRIX_H_

#include <deal.II/lac/lapack_full_matrix.h>

#include "matrix_helper.h"
#include "move_to_deal_ii.h"
#include "tensors.h"

using namespace dealii;

namespace Tensors
{
template<int order, typename Number, int n_rows_1d = -1>
class TensorProductMatrix : public TensorProductMatrixSymmetricSum<order, Number, n_rows_1d>
{
public:
  enum class State
  {
    invalid,
    basic,
    skd
  };

  using SKDMatrix         = TensorProductMatrixSymmetricSum<order, Number, n_rows_1d>;
  using value_type        = Number;
  using scalar_value_type = typename ExtractScalarType<Number>::type;
  using matrix_type_1d    = Table<2, Number>;

  TensorProductMatrix() = default;

  TensorProductMatrix(const std::vector<std::array<Table<2, Number>, order>> & elementary_tensors,
                      const State state_in = State::basic)
  {
    reinit(elementary_tensors, state_in);
  }

  TensorProductMatrix &
  operator=(const TensorProductMatrix & other)
  {
    reinit(other.elementary_tensors, other.state);
    return *this;
  }

  void
  clear()
  {
    tmp_array.clear();
    basic_inverse.reset();
    state = State::invalid;
    elementary_tensors.clear();
    /// TODO clear underlying TensorProductMatrixSymmetricSum
  }

  void
  reinit(const std::vector<std::array<Table<2, Number>, order>> & elementary_tensors_in,
         const State                                              state_in = State::basic)
  {
    Assert(check_static_n_rows_1d(elementary_tensors_in),
           ExcMessage("Not all univariate matrices are of size (n_rows_1d x n_rows_1d)."));

    /// clear old data
    clear();
    if(elementary_tensors_in.empty())
      return;

    /// initialize new data
    state = state_in;

    if(state == State::basic)
    {
      for(unsigned i = 0; i < order; ++i)
        Assert(check_size_1d(elementary_tensors_in, i),
               ExcMessage("Mismatching sizes of univariate matrices."));
      std::copy(elementary_tensors_in.cbegin(),
                elementary_tensors_in.cend(),
                std::back_inserter(elementary_tensors));

      const auto &                    first_elementary_tensor = elementary_tensors.front();
      std::array<unsigned int, order> n_rows_foreach_dimension;
      std::array<unsigned int, order> n_columns_foreach_dimension;
      for(auto d = 0; d < order; ++d)
      {
        const auto & matrix            = first_elementary_tensor[d];
        n_rows_foreach_dimension[d]    = matrix.size(0);
        n_columns_foreach_dimension[d] = matrix.size(1);
      }
      tensor_helper_row = std::make_shared<const TensorHelper<order>>(n_rows_foreach_dimension);
      tensor_helper_column =
        std::make_shared<const TensorHelper<order>>(n_columns_foreach_dimension);
    }

    else if(state == State::skd)
    {
      AssertThrow(
        elementary_tensors_in.size() == 2,
        ExcMessage(
          "Two tensors are required, namely a tensor of mass matrices and a tensor of derivative matrices."));
      /// TODO avoid duplication
      // elementary_tensor = elementary_tensor_in;
      SKDMatrix::reinit(elementary_tensors_in[0], elementary_tensors_in[1]);
    }

    else
      AssertThrow(false, ExcMessage("Invalid state at initialization."));
  }

  AlignedVector<Number>
  get_eigenvalues() const
  {
    AssertThrow(state == State::skd, ExcMessage("Not implemented."));
    AlignedVector<Number>           eigenvalues(m());
    const auto &                    evs_1d = SKDMatrix::eigenvalues;
    std::array<unsigned int, order> sizes;
    std::transform(evs_1d.cbegin(), evs_1d.cend(), sizes.begin(), [](const auto & evs) {
      return evs.size();
    });
    for(unsigned int i = 0; i < eigenvalues.size(); ++i)
    {
      const auto & ii     = uni_to_multiindex<order>(i, sizes);
      Number       lambda = evs_1d[0][ii[0]];
      for(auto d = 1; d < order; ++d)
        lambda += evs_1d[d][ii[d]];
      eigenvalues[i] = lambda;
    }
    // TODO
    const bool has_zero_eigenvalues =
      std::any_of(eigenvalues.begin(), eigenvalues.end(), Tensors::is_nearly_zero_value<Number>);
    AssertThrow(!has_zero_eigenvalues, ExcMessage("Has zero eigenvalues."));

    return eigenvalues;
  }

  const std::array<Table<2, Number>, order> &
  get_mass() const
  {
    AssertThrow(state == State::skd, ExcMessage("Not implemented."));
    return this->mass_matrix;
  }

  const std::array<Table<2, Number>, order> &
  get_eigenvectors() const
  {
    AssertThrow(state == State::skd, ExcMessage("Not implemented."));
    return this->eigenvectors;
  }

  std::vector<std::array<Table<2, Number>, order>>
  get_elementary_tensors() const
  {
    if(state == State::skd)
    {
      std::vector<std::array<Table<2, Number>, order>> tensors(order);
      const auto &                                     mass       = this->mass_matrix;
      const auto &                                     derivative = this->derivative_matrix;
      for(auto i = 0U; i < order; ++i)
      {
        auto & tensor = tensors[i];
        for(auto j = 0U; j < order; ++j)
          tensor[j] = i == j ? derivative[j] : mass[j];
      }
      return tensors;
    }
    Assert(state != State::invalid, ExcMessage("Invalid State"));
    return elementary_tensors;
  }

  State
  get_state() const
  {
    return state;
  }

  unsigned int
  m() const
  {
    if(state == State::skd)
      return SKDMatrix::m();
    Assert(elementary_tensors.size() > 0, ExcMessage("Not initialized."));
    const unsigned int m_left  = left(0).size(0);
    const unsigned int m_right = right(0).size(0);
    return m_left * m_right;
  }

  unsigned int
  n() const
  {
    if(state == State::skd)
      return SKDMatrix::n();
    Assert(elementary_tensors.size() > 0, ExcMessage("Not initialized."));
    const unsigned int n_left  = left(0).size(1);
    const unsigned int n_right = right(0).size(1);
    return n_left * n_right;
  }

  unsigned int
  m(unsigned int dimension) const
  {
    AssertIndexRange(dimension, order);
    if(state == State::skd)
      return SKDMatrix::eigenvalues[dimension].size();
    Assert(elementary_tensors.size() > 0, ExcMessage("Not initialized."));
    return elementary_tensors.front()[dimension].n_rows();
  }

  unsigned int
  n(unsigned int dimension) const
  {
    AssertIndexRange(dimension, order);
    if(state == State::skd)
      return SKDMatrix::eigenvalues[dimension].size();
    Assert(elementary_tensors.size() > 0, ExcMessage("Not initialized."));
    return elementary_tensors.front()[dimension].n_cols();
  }

  void
  vmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    vmult_impl</*add*/ false>(dst_view, src_view);
  }

  void
  vmult_add(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    vmult_impl</*add*/ true>(dst_view, src_view);
  }

  void
  Tvmult(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    vmult_impl</*add*/ false, /*transpose*/ true>(dst_view, src_view);
  }

  void
  apply_inverse(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    apply_inverse_impl(dst_view, src_view);
  }

  Table<2, Number>
  as_table() const
  {
    return Tensors::matrix_to_table(*this);
  }

  Table<2, Number>
  as_inverse_table() const
  {
    return Tensors::inverse_matrix_to_table(*this);
  }

  Table<2, Number>
  as_transpose_table() const
  {
    return Tensors::transpose_matrix_to_table(*this);
  }

  // void
  // copy_from (TensorProductMatrix<order, VectorizedArray<scalar_value_type>, n_rows_1d> & other,
  // const unsigned int lane = 0)
  // {
  //   static_assert(std::is_same<Number, scalar_value_type>::value, "Copy is only valid if the
  //   underlying class is of scalar value type"); AssertIndexRange(lane,
  //   VectorizedArray<scalar_value_type>::size()); std::vector<std::array<Table<2,
  //   scalar_value_type>, order>> tensors_lane; const auto & other_tensors =
  //   other.elementary_tensors;
  // }

protected:
  const Table<2, Number> &
  left(unsigned int r) const
  {
    static_assert(order == 2, "TODO");
    AssertIndexRange(r, elementary_tensors.size());
    return elementary_tensors[r][1];
  }

  const Table<2, Number> &
  right(unsigned int r) const
  {
    static_assert(order == 2, "TODO");
    AssertIndexRange(r, elementary_tensors.size());
    return elementary_tensors[r][0];
  }

  /**
   * Vector containing all elementary tensors, where each tensor consists of @p
   * order-1 Kronecker products. If linear independent the vector size
   * determines the tensor rank (or Kronecker rank).
   *
   * NOTE Elementary tensors are only stored in this field for the basic
   * state. For the skd state the underlying TensorProductMatrixSymmetricSum
   * stores the tensors.
   */
  std::vector<std::array<Table<2, Number>, order>> elementary_tensors;

  /**
   * The state switches which functionalities are faciliated:
   *
   * basic:  TensorProductMatrix
   * skd:    TensorProductMatrixSymmetricSum
   */
  State state = State::invalid;

private:
  void
  apply_inverse_impl(const ArrayView<Number> &       dst_view,
                     const ArrayView<const Number> & src_view) const
  {
    if(state == State::basic)
      apply_inverse_impl_basic_static(dst_view, src_view);
    else if(state == State::skd)
      SKDMatrix::apply_inverse(dst_view, src_view);
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }

  void
  apply_inverse_impl_basic_static(const ArrayView<Number> &       dst_view,
                                  const ArrayView<const Number> & src_view) const
  {
    AssertDimension(dst_view.size(), m());
    AssertDimension(src_view.size(), n());
    if(!basic_inverse)
      basic_inverse = std::make_shared<const InverseTable<Number>>(as_table());
    basic_inverse->vmult(dst_view, src_view);
  }

  template<bool add, bool transpose = false>
  void
  vmult_impl(const ArrayView<Number> & dst_view, const ArrayView<const Number> & src_view) const
  {
    if(state == State::basic)
      vmult_impl_basic_static<add, transpose>(dst_view, src_view);
    else if(state == State::skd)
    {
      AssertThrow(!transpose, ExcMessage("Not implemented."));
      AlignedVector<Number> initial_dst;
      if(add)
      {
        initial_dst.resize_fast(dst_view.size());
        std::copy(dst_view.cbegin(), dst_view.cend(), initial_dst.begin());
      }
      SKDMatrix::vmult(dst_view, src_view);
      if(add)
      {
        std::transform(dst_view.cbegin(),
                       dst_view.cend(),
                       initial_dst.begin(),
                       dst_view.begin(),
                       [](const auto & elem1, const auto & elem2) { return elem1 + elem2; });
      }
    }
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }


  template<bool add, bool transpose = false>
  void
  vmult_impl_basic_static(const ArrayView<Number> &       dst_view,
                          const ArrayView<const Number> & src_view) const
  {
    static_assert(order == 2, "TODO");
    AssertDimension(dst_view.size(), m());
    AssertDimension(src_view.size(), n());
    AssertDimension(left(0).size(0), right(0).size(0)); // assuming isotropy
    AssertDimension(left(0).size(1), right(0).size(1)); // assuming isotropy
    Assert(tensor_helper_row, ExcMessage("Did you initialize tensor_helper_row?"));
    Assert(tensor_helper_column, ExcMessage("Did you initialize tensor_helper_column?"));

    constexpr bool is_dynamic_path = n_rows_1d == -1;

    constexpr int mm_static = Utilities::pow(n_rows_1d, order); // TODO anisotropic

    const unsigned int mm_dynamic =
      transpose ? tensor_helper_column->n_flat() : tensor_helper_row->n_flat(); // TODO check
    const int n_rows_1d_dynamic    = tensor_helper_row->size(0);                // TODO anisotropic
    const int n_columns_1d_dynamic = tensor_helper_column->size(0);             // TODO anisotropic

    const unsigned int mm = is_dynamic_path ? mm_dynamic : mm_static;

    std::lock_guard<std::mutex> lock(this->mutex);
    tmp_array.clear();
    tmp_array.resize(mm);
    Number *       tmp = tmp_array.begin();
    const Number * src = src_view.begin();
    Number *       dst = dst_view.data();

    using Evaluator =
      dealii::internal::My::EvaluatorTensorProduct<order, Number, n_rows_1d, n_rows_1d>;
    Evaluator eval(n_rows_1d_dynamic, n_columns_1d_dynamic);

    /// Apply elementary tensor of first rank.
    const Number * left_0  = &(left(0)(0, 0));
    const Number * right_0 = &(right(0)(0, 0));
    if(transpose)
    {
      eval.template apply<1, transpose, false>(left_0, src, tmp);
      eval.template apply<0, transpose, add>(right_0, tmp, dst);
    }
    else
    {
      eval.template apply<0, transpose, false>(right_0, src, tmp);
      eval.template apply<1, transpose, add>(left_0, tmp, dst);
    }

    /// Apply elementary tensors of remaining ranks.
    for(std::size_t r = 1; r < elementary_tensors.size(); ++r)
    {
      const Number * left_r  = &(left(r)(0, 0));
      const Number * right_r = &(right(r)(0, 0));
      if(transpose)
      {
        eval.template apply<1, transpose, false>(left_r, src, tmp);
        eval.template apply<0, transpose, true>(right_r, tmp, dst);
      }
      else
      {
        eval.template apply<0, transpose, false>(right_r, src, tmp);
        eval.template apply<1, transpose, true>(left_r, tmp, dst);
      }
    }
  }

  bool
  check_static_n_rows_1d(const std::vector<std::array<Table<2, Number>, order>> & tensors)
  {
    if(n_rows_1d == -1)
      return true;
    if(n_rows_1d > 0)
    {
      std::vector<unsigned int> indices(order);
      std::iota(indices.begin(), indices.end(), 0);
      return std::all_of(indices.cbegin(), indices.cend(), [&](const auto & i) {
        return check_size_1d_impl(tensors, i, n_rows_1d, n_rows_1d);
      });
    }
    return false;
  }

  bool
  check_size_1d(const std::vector<std::array<Table<2, Number>, order>> & tensors,
                const unsigned int                                       tensor_index)
  {
    Assert(!tensors.empty(), ExcMessage("No tensors provided."));
    const unsigned int n_rows = tensors.front()[tensor_index].size(0);
    const unsigned int n_cols = tensors.front()[tensor_index].size(1);
    return check_size_1d_impl(tensors, tensor_index, n_rows, n_cols);
  }

  bool
  check_size_1d_impl(const std::vector<std::array<Table<2, Number>, order>> & tensors,
                     const unsigned int                                       tensor_index,
                     const unsigned int                                       n_rows,
                     const unsigned int                                       n_cols)
  {
    return std::all_of(tensors.cbegin(),
                       tensors.cend(),
                       [tensor_index, n_rows, n_cols](const auto & tensor) {
                         const auto & tab = tensor[tensor_index];
                         return tab.size(0) == n_rows && tab.size(1) == n_cols;
                       });
  }

  std::shared_ptr<const TensorHelper<order>> tensor_helper_row;
  std::shared_ptr<const TensorHelper<order>> tensor_helper_column;

  /**
   * The naive inverse of the underlying matrix for the basic state.
   */
  mutable std::shared_ptr<const InverseTable<Number>> basic_inverse;

  /**
   * A mutex that guards access to the array @p tmp_array.
   */
  mutable Threads::Mutex mutex;

  /**
   * An array for temporary data.
   */
  mutable AlignedVector<Number> tmp_array;
};

} // namespace Tensors

#endif // TENSOR_PRODUCT_MATRIX_H_
