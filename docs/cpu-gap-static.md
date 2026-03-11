# Static CPU Gap Report (Candle vs PyTorch)

Date: 2026-03-11

## Method
- Source: PyTorch `aten/src/ATen/native/native_functions.yaml` (v2.5.0 via compat clone).
- Candle side: schema registry after `register_schemas()` (covers all op schemas registered in `src/candle/_dispatch/schemas.py`).
- CPU-eligible ops: PyTorch entries with `dispatch` containing `CPU` or a `Composite*` key.
- Exclusions: internal names (leading `_`) and codegen placeholder names (Scalar/Tensor/List/etc.).
- This report is a static interface gap list (not correctness gaps).

## Summary
- PyTorch CPU-eligible ops: 647
- Candle schema ops: 384
- Missing ops (filtered): 267
- Missing ops (forward-only): 211

## Missing Ops (Forward-Only)

### view (8)
- `as_strided_`
- `as_strided_copy`
- `as_strided_scatter`
- `expand_copy`
- `slice`
- `slice_copy`
- `slice_inverse`
- `slice_scatter`

### reduction (1)
- `sum_to_size`

### binary (7)
- `addmv`
- `addr`
- `addr_`
- `floor_divide_`
- `fmod_`
- `multi_margin_loss`
- `remainder_`

### unary (8)
- `abs_`
- `fractional_max_pool2d`
- `fractional_max_pool3d`
- `logcumsumexp`
- `logical_and_`
- `logical_not_`
- `logical_or_`
- `logical_xor_`

### pooling (2)
- `max_pool2d_with_indices`
- `max_pool3d_with_indices`

### conv (5)
- `conv_tbc`
- `convolution`
- `convolution_backward_overrideable`
- `convolution_overrideable`
- `mkldnn_convolution`

### fft (6)
- `fft_fftfreq`
- `fft_hfft2`
- `fft_hfftn`
- `fft_ihfft2`
- `fft_ihfftn`
- `fft_rfftfreq`

### special (12)
- `special_chebyshev_polynomial_t`
- `special_chebyshev_polynomial_u`
- `special_chebyshev_polynomial_v`
- `special_chebyshev_polynomial_w`
- `special_hermite_polynomial_h`
- `special_hermite_polynomial_he`
- `special_laguerre_polynomial_l`
- `special_legendre_polynomial_p`
- `special_shifted_chebyshev_polynomial_t`
- `special_shifted_chebyshev_polynomial_u`
- `special_shifted_chebyshev_polynomial_v`
- `special_shifted_chebyshev_polynomial_w`

### random (7)
- `bernoulli`
- `binomial`
- `normal`
- `normal_functional`
- `poisson`
- `rand_like`
- `randint_like`

### loss (5)
- `cross_entropy_loss`
- `nll_loss`
- `nll_loss2d`
- `nll_loss_nd`
- `soft_margin_loss`

### other (150)
- `affine_grid_generator`
- `alias`
- `alias_copy`
- `bartlett_window`
- `batch_norm_update_stats`
- `binary_cross_entropy`
- `binary_cross_entropy_with_logits`
- `bitwise_and_`
- `bitwise_left_shift`
- `bitwise_left_shift_`
- `bitwise_or_`
- `bitwise_right_shift`
- `bitwise_right_shift_`
- `bitwise_xor_`
- `blackman_window`
- `ccol_indices`
- `ccol_indices_copy`
- `celu_`
- `cholesky_solve`
- `clone`
- `col_indices`
- `col_indices_copy`
- `complex`
- `conj_physical_`
- `constant_pad_nd`
- `copy`
- `copysign`
- `copysign_`
- `crow_indices`
- `crow_indices_copy`
- `deg2rad`
- `deg2rad_`
- `dense_dim`
- `detach`
- `detach_`
- `detach_copy`
- `diag_embed`
- `diagonal_copy`
- `diagonal_scatter`
- `embedding_renorm_`
- `empty_like`
- `empty_permuted`
- `empty_strided`
- `fill`
- `frexp`
- `from_file`
- `full_like`
- `grid_sampler_3d`
- `hamming_window`
- `hann_window`
- `index_add`
- `index_fill`
- `index_reduce`
- `indices`
- `indices_copy`
- `is_coalesced`
- `is_same_size`
- `kaiser_window`
- `lift`
- `lift_fresh`
- `lift_fresh_copy`
- `linear`
- `masked_scatter`
- `mkldnn_rnn_layer`
- `mode`
- `mvlgamma`
- `mvlgamma_`
- `nan_to_num`
- `nan_to_num_`
- `narrow_copy`
- `native_batch_norm`
- `native_channel_shuffle`
- `native_dropout`
- `native_group_norm`
- `native_layer_norm`
- `new_empty`
- `new_empty_strided`
- `new_full`
- `new_ones`
- `new_zeros`
- `nonzero_static`
- `ones_like`
- `permute_copy`
- `pixel_shuffle`
- `pixel_unshuffle`
- `polar`
- `polygamma_`
- `put`
- `quantize_per_tensor`
- `rad2deg`
- `rad2deg_`
- `reflection_pad1d`
- `reflection_pad2d`
- `reflection_pad3d`
- `replication_pad1d`
- `replication_pad2d`
- `replication_pad3d`
- `reshape_as`
- `resize_`
- `resize_as_`
- `row_indices`
- `row_indices_copy`
- `rrelu_with_noise`
- `rrelu_with_noise_`
- `rsub`
- `scalar_tensor`
- `select_copy`
- `select_scatter`
- `set_`
- `slow_conv_dilated2d`
- `slow_conv_dilated3d`
- `slow_conv_transpose2d`
- `slow_conv_transpose3d`
- `sparse_compressed_tensor`
- `sparse_coo_tensor`
- `sparse_dim`
- `split_copy`
- `split_with_sizes`
- `split_with_sizes_copy`
- `squeeze_`
- `squeeze_copy`
- `sspaddmm`
- `sym_constrain_range`
- `sym_constrain_range_for_size`
- `t`
- `t_`
- `t_copy`
- `tensor_split`
- `to_mkldnn`
- `transpose_`
- `transpose_copy`
- `unbind_copy`
- `unfold_copy`
- `unique_consecutive`
- `unique_dim`
- `unique_dim_consecutive`
- `unsafe_split`
- `unsafe_split_with_sizes`
- `unsqueeze_`
- `unsqueeze_copy`
- `upsample_nearest3d`
- `upsample_trilinear3d`
- `values`
- `values_copy`
- `vdot`
- `view_as_complex_copy`
- `view_as_real_copy`
- `view_copy`
- `xlogy`
- `xlogy_`

## Missing Ops (All, Including Backward/Forward Helpers)

### view (9)
- `as_strided_`
- `as_strided_copy`
- `as_strided_scatter`
- `expand_copy`
- `slice`
- `slice_backward`
- `slice_copy`
- `slice_inverse`
- `slice_scatter`

### reduction (1)
- `sum_to_size`

### binary (10)
- `addmv`
- `addr`
- `addr_`
- `floor_divide_`
- `fmod_`
- `multi_margin_loss`
- `multi_margin_loss_backward`
- `multilabel_margin_loss_backward`
- `multilabel_margin_loss_forward`
- `remainder_`

### unary (15)
- `abs_`
- `fractional_max_pool2d`
- `fractional_max_pool2d_backward`
- `fractional_max_pool3d`
- `fractional_max_pool3d_backward`
- `gelu_backward`
- `log_sigmoid_backward`
- `log_sigmoid_forward`
- `logcumsumexp`
- `logical_and_`
- `logical_not_`
- `logical_or_`
- `logical_xor_`
- `mish_backward`
- `silu_backward`

### pooling (9)
- `adaptive_avg_pool3d_backward`
- `adaptive_max_pool2d_backward`
- `adaptive_max_pool3d_backward`
- `avg_pool2d_backward`
- `avg_pool3d_backward`
- `max_pool2d_with_indices`
- `max_pool2d_with_indices_backward`
- `max_pool3d_with_indices`
- `max_pool3d_with_indices_backward`

### conv (5)
- `conv_tbc`
- `convolution`
- `convolution_backward_overrideable`
- `convolution_overrideable`
- `mkldnn_convolution`

### fft (6)
- `fft_fftfreq`
- `fft_hfft2`
- `fft_hfftn`
- `fft_ihfft2`
- `fft_ihfftn`
- `fft_rfftfreq`

### special (12)
- `special_chebyshev_polynomial_t`
- `special_chebyshev_polynomial_u`
- `special_chebyshev_polynomial_v`
- `special_chebyshev_polynomial_w`
- `special_hermite_polynomial_h`
- `special_hermite_polynomial_he`
- `special_laguerre_polynomial_l`
- `special_legendre_polynomial_p`
- `special_shifted_chebyshev_polynomial_t`
- `special_shifted_chebyshev_polynomial_u`
- `special_shifted_chebyshev_polynomial_v`
- `special_shifted_chebyshev_polynomial_w`

### random (7)
- `bernoulli`
- `binomial`
- `normal`
- `normal_functional`
- `poisson`
- `rand_like`
- `randint_like`

### loss (12)
- `cross_entropy_loss`
- `huber_loss_backward`
- `nll_loss`
- `nll_loss2d`
- `nll_loss2d_backward`
- `nll_loss2d_forward`
- `nll_loss_backward`
- `nll_loss_forward`
- `nll_loss_nd`
- `smooth_l1_loss_backward`
- `soft_margin_loss`
- `soft_margin_loss_backward`

### other (181)
- `affine_grid_generator`
- `alias`
- `alias_copy`
- `bartlett_window`
- `batch_norm_backward`
- `batch_norm_update_stats`
- `binary_cross_entropy`
- `binary_cross_entropy_backward`
- `binary_cross_entropy_with_logits`
- `bitwise_and_`
- `bitwise_left_shift`
- `bitwise_left_shift_`
- `bitwise_or_`
- `bitwise_right_shift`
- `bitwise_right_shift_`
- `bitwise_xor_`
- `blackman_window`
- `ccol_indices`
- `ccol_indices_copy`
- `celu_`
- `cholesky_solve`
- `clone`
- `col_indices`
- `col_indices_copy`
- `complex`
- `conj_physical_`
- `constant_pad_nd`
- `copy`
- `copysign`
- `copysign_`
- `crow_indices`
- `crow_indices_copy`
- `deg2rad`
- `deg2rad_`
- `dense_dim`
- `detach`
- `detach_`
- `detach_copy`
- `diag_embed`
- `diagonal_backward`
- `diagonal_copy`
- `diagonal_scatter`
- `embedding_backward`
- `embedding_dense_backward`
- `embedding_renorm_`
- `empty_like`
- `empty_permuted`
- `empty_strided`
- `fill`
- `frexp`
- `from_file`
- `full_like`
- `glu_backward`
- `grid_sampler_2d_backward`
- `grid_sampler_3d`
- `grid_sampler_3d_backward`
- `hamming_window`
- `hann_window`
- `index_add`
- `index_fill`
- `index_reduce`
- `index_select_backward`
- `indices`
- `indices_copy`
- `is_coalesced`
- `is_same_size`
- `kaiser_window`
- `lift`
- `lift_fresh`
- `lift_fresh_copy`
- `linear`
- `masked_scatter`
- `masked_scatter_backward`
- `mkldnn_rnn_layer`
- `mkldnn_rnn_layer_backward`
- `mode`
- `mvlgamma`
- `mvlgamma_`
- `nan_to_num`
- `nan_to_num_`
- `narrow_copy`
- `native_batch_norm`
- `native_batch_norm_backward`
- `native_channel_shuffle`
- `native_dropout`
- `native_group_norm`
- `native_layer_norm`
- `native_layer_norm_backward`
- `new_empty`
- `new_empty_strided`
- `new_full`
- `new_ones`
- `new_zeros`
- `nonzero_static`
- `ones_like`
- `permute_copy`
- `pixel_shuffle`
- `pixel_unshuffle`
- `polar`
- `polygamma_`
- `put`
- `quantize_per_tensor`
- `rad2deg`
- `rad2deg_`
- `reflection_pad1d`
- `reflection_pad1d_backward`
- `reflection_pad2d`
- `reflection_pad2d_backward`
- `reflection_pad3d`
- `reflection_pad3d_backward`
- `replication_pad1d`
- `replication_pad1d_backward`
- `replication_pad2d`
- `replication_pad2d_backward`
- `replication_pad3d`
- `replication_pad3d_backward`
- `reshape_as`
- `resize_`
- `resize_as_`
- `row_indices`
- `row_indices_copy`
- `rrelu_with_noise`
- `rrelu_with_noise_`
- `rrelu_with_noise_backward`
- `rsub`
- `scalar_tensor`
- `select_backward`
- `select_copy`
- `select_scatter`
- `set_`
- `slow_conv3d_forward`
- `slow_conv_dilated2d`
- `slow_conv_dilated3d`
- `slow_conv_transpose2d`
- `slow_conv_transpose3d`
- `sparse_compressed_tensor`
- `sparse_coo_tensor`
- `sparse_dim`
- `split_copy`
- `split_with_sizes`
- `split_with_sizes_copy`
- `squeeze_`
- `squeeze_copy`
- `sspaddmm`
- `sym_constrain_range`
- `sym_constrain_range_for_size`
- `t`
- `t_`
- `t_copy`
- `tensor_split`
- `to_mkldnn`
- `trace_backward`
- `transpose_`
- `transpose_copy`
- `unbind_copy`
- `unfold_copy`
- `unique_consecutive`
- `unique_dim`
- `unique_dim_consecutive`
- `unsafe_split`
- `unsafe_split_with_sizes`
- `unsqueeze_`
- `unsqueeze_copy`
- `upsample_bicubic2d_backward`
- `upsample_bilinear2d_backward`
- `upsample_linear1d_backward`
- `upsample_nearest1d_backward`
- `upsample_nearest2d_backward`
- `upsample_nearest3d`
- `upsample_nearest3d_backward`
- `upsample_trilinear3d`
- `upsample_trilinear3d_backward`
- `value_selecting_reduction_backward`
- `values`
- `values_copy`
- `vdot`
- `view_as_complex_copy`
- `view_as_real_copy`
- `view_copy`
- `xlogy`
- `xlogy_`
