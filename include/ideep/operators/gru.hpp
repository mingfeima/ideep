#ifndef IDEEP_OPERATORS_GRU_HPP
#define IDEEP_OPERATORS_GRU_HPP

namespace ideep {

struct gru_forward_inference : public dnnl::lbr_gru_forward {

  using super = dnnl::lbr_gru_forward ;

  static void compute(const tensor& src_layer,
                      const tensor& src_iter,
                      const tensor& weights_layer,
                      const tensor& weights_iter,
                      const tensor& bias,
                      tensor& dst_layer,
                      tensor& dst_iter,
                      const bool reverse = false,
                      const engine& aengine = engine::cpu_engine()) {
    auto aprop = prop_kind::forward_inference;
    auto direction = reverse ? rnn_direction::unidirectional_right2left
                             : rnn_direction::unidirectional_left2right;

    auto src_layer_desc = src_layer.get_desc();
    auto src_iter_desc = src_iter.get_desc();
    // use any format for weights
    auto weights_layer_desc = weights_layer.get_desc().to_format_any();
    auto weights_iter_desc = weights_iter.get_desc().to_format_any();
    auto bias_desc = bias.get_desc();
    auto dst_layer_desc = dst_layer.get_desc();
    auto dst_iter_desc = dst_iter.get_desc();

    auto pd = primitive_desc(
        {aprop, direction, src_layer_desc, src_iter_desc,
         weights_layer_desc, weights_iter_desc, bias_desc,
         dst_layer_desc, dst_iter_desc},
        aengine);

    auto expected_weights_layer = weights_layer.reorder_if_differ_in(pd.weights_desc());
    auto expected_weights_iter = weights_iter.reorder_if_differ_in(pd.weights_iter_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC_LAYER, src_layer},
                       {DNNL_ARG_SRC_ITER, src_iter},
                       {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
                       {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
                       {DNNL_ARG_BIAS, bias},
                       {DNNL_ARG_DST_LAYER, dst_layer},
                       {DNNL_ARG_DST_ITER, dst_iter}});
  }
};

struct gru_backward : public dnnl::lbr_gru_backward {
  static void compute() {
  }
};

}  // namespace ideep

#endif