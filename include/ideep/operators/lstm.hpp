#ifndef IDEEP_OPERATORS_LSTM_HPP
#define IDEEP_OPERATORS_LSTM_HPP

namespace ideep {

struct lstm_forward : public dnnl::lstm_forward {

  using super = dnnl::lstm_forward;

  template <bool train>
  static primitive_desc get_primitive_desc(
      const tensor::desc& src_layer_desc,
      const tensor::desc& src_iter_desc,
      const tensor::desc& src_iter_c_desc,
      const tensor::desc& weights_layer_desc,
      const tensor::desc& weights_iter_desc,
      const tensor::desc& bias_desc,
      const tensor::desc& dst_layer_desc,
      const tensor::desc& dst_iter_desc,
      const tensor::desc& dst_iter_c_desc,
      const bool reverse,
      const engine& aengine = engine::cpu_engine()) {
    auto aprop = train ? prop_kind::forward_training
                       : prop_kind::forward_inference;
    auto direction = reverse ? rnn_direction::unidirectional_right2left
                             : rnn_direction::unidirectional_left2right;
    // use any format for weights
    auto weights_layer_desc_any = weights_layer_desc.to_format_any();
    auto weights_iter_desc_any = weights_iter_desc.to_format_any();
    auto bias_desc_any = bias_desc.to_format_any();

    return primitive_desc({aprop, direction, src_layer_desc, src_iter_desc, src_iter_c_desc,
                           weights_layer_desc_any, weights_iter_desc_any, bias_desc_any,
                           dst_layer_desc, dst_iter_desc, dst_iter_c_desc},
                          aengine);
  }

  static tensor::desc get_workspace_desc(
      const tensor::desc& src_layer_desc,
      const tensor::desc& src_iter_desc,
      const tensor::desc& src_iter_c_desc,
      const tensor::desc& weights_layer_desc,
      const tensor::desc& weights_iter_desc,
      const tensor::desc& bias_desc,
      const tensor::desc& dst_layer_desc,
      const tensor::desc& dst_iter_desc,
      const tensor::desc& dst_iter_c_desc,
      const bool reverse) {
    auto pd = get_primitive_desc</*train*/true>(
        src_layer_desc, src_iter_desc, src_iter_c_desc,
        weights_layer_desc, weights_iter_desc, bias_desc,
        dst_layer_desc, dst_iter_desc, dst_iter_c_desc,
        reverse);

    return {pd.workspace_desc()};
  }

  static void compute(
      const tensor& src_layer,
      const tensor& src_iter,
      const tensor& src_iter_c,
      const tensor& weights_layer,
      const tensor& weights_iter,
      const tensor& bias,
      tensor& dst_layer,
      tensor& dst_iter,
      tensor& dst_iter_c,
      const bool reverse = false,
      const engine& aengine = engine::cpu_engine()) {

    auto pd = get_primitive_desc</*train*/false>(
        src_layer.get_desc(), src_iter.get_desc(), src_iter_c.get_desc(),
        weights_layer.get_desc(), weights_iter.get_desc(), bias.get_desc(),
        dst_layer.get_desc(), dst_iter.get_desc(), dst_iter_c.get_desc(),
        reverse);

    auto expected_weights_layer = weights_layer.reorder_if_differ_in(pd.weights_desc());
    auto expected_weights_iter = weights_iter.reorder_if_differ_in(pd.weights_iter_desc());
    auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC_LAYER, src_layer},
                       {DNNL_ARG_SRC_ITER, src_iter},
                       {DNNL_ARG_SRC_ITER_C, src_iter_c},
                       {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
                       {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
                       {DNNL_ARG_BIAS, expected_bias},
                       {DNNL_ARG_DST_LAYER, dst_layer},
                       {DNNL_ARG_DST_ITER, dst_iter},
                       {DNNL_ARG_DST_ITER_C, dst_iter_c}});
  }

  static void compute(
      const tensor& src_layer,
      const tensor& src_iter,
      const tensor& src_iter_c,
      const tensor& weights_layer,
      const tensor& weights_iter,
      const tensor& bias,
      tensor& dst_layer,
      tensor& dst_iter,
      tensor& dst_iter_c,
      tensor& workspace,
      const bool reverse = false,
      const engine& aengine = engine::cpu_engine()) {

    auto pd = get_primitive_desc</*train*/true>(
        src_layer.get_desc(), src_iter.get_desc(), src_iter_c.get_desc(),
        weights_layer.get_desc(), weights_iter.get_desc(), bias.get_desc(),
        dst_layer.get_desc(), dst_iter.get_desc(), dst_iter_c.get_desc(),
        reverse);

    auto expected_weights_layer = weights_layer.reorder_if_differ_in(pd.weights_desc());
    auto expected_weights_iter = weights_iter.reorder_if_differ_in(pd.weights_iter_desc());
    auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC_LAYER, src_layer},
                       {DNNL_ARG_SRC_ITER, src_iter},
                       {DNNL_ARG_SRC_ITER_C, src_iter_c},
                       {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
                       {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
                       {DNNL_ARG_BIAS, expected_bias},
                       {DNNL_ARG_DST_LAYER, dst_layer},
                       {DNNL_ARG_DST_ITER, dst_iter},
                       {DNNL_ARG_DST_ITER_C, dst_iter_c},
                       {DNNL_ARG_WORKSPACE, workspace}});
  }
};

struct lstm_backward : public dnnl::lstm_backward {

  using super = dnnl::lstm_backward;

  static void compute(
      const tensor& src_layer,
      const tensor& src_iter,
      const tensor& src_iter_c,
      const tensor& weights_layer,
      const tensor& weights_iter,
      const tensor& bias,
      const tensor& dst_layer,
      const tensor& dst_iter,
      const tensor& dst_iter_c,
      tensor& diff_src_layer,
      tensor& diff_src_iter,
      tensor& diff_src_iter_c,
      tensor& diff_weights_layer,
      tensor& diff_weights_iter,
      tensor& diff_bias,
      const tensor& diff_dst_layer,
      const tensor& diff_dst_iter,
      const tensor& diff_dst_iter_c,
      const tensor& workspace,
      const bool reverse = false,
      const engine& aengine = engine::cpu_engine()) {

    auto aprop = prop_kind::backward;
    auto direction = reverse ? rnn_direction::unidirectional_right2left
                             : rnn_direction::unidirectional_left2right;

    // use any formats for weights and diff_weights
    auto weights_layer_desc_any = weights_layer.get_desc().to_format_any();
    auto weights_iter_desc_any = weights_iter.get_desc().to_format_any();
    auto bias_desc_any = bias.get_desc().to_format_any();
    auto diff_weights_layer_desc_any = diff_weights_layer.get_desc().to_format_any();
    auto diff_weights_iter_desc_any = diff_weights_iter.get_desc().to_format_any();
    auto diff_bias_desc_any = diff_bias.get_desc().to_format_any();

    auto forward_hints =
        lstm_forward::get_primitive_desc</*train*/true>(
            src_layer.get_desc(), src_iter.get_desc(), src_iter_c.get_desc(),
            weights_layer.get_desc(), weights_iter.get_desc(), bias.get_desc(),
            dst_layer.get_desc(), dst_iter.get_desc(), dst_iter_c.get_desc(),
            reverse);

    auto pd = primitive_desc(
        {aprop, direction, src_layer.get_desc(), src_iter.get_desc(), src_iter_c.get_desc(),
         weights_layer_desc_any, weights_iter_desc_any, bias_desc_any,
         dst_layer.get_desc(), dst_iter.get_desc(), dst_iter_c.get_desc(),
         diff_src_layer.get_desc(), diff_src_iter.get_desc(), diff_src_iter_c.get_desc(),
         diff_weights_layer_desc_any, diff_weights_iter_desc_any, diff_bias_desc_any,
         diff_dst_layer.get_desc(), diff_dst_iter.get_desc(), diff_dst_iter_c.get_desc()},
        aengine, forward_hints);

    auto expected_weights_layer = weights_layer.reorder_if_differ_in(pd.weights_desc());
    auto expected_weights_iter = weights_iter.reorder_if_differ_in(pd.weights_iter_desc());
    auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc());
    auto expected_diff_weights_layer = tensor(pd.diff_weights_layer_desc());
    auto expected_diff_weights_iter = tensor(pd.diff_weights_iter_desc());
    auto expected_diff_bias = tensor(pd.diff_bias_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC_LAYER, src_layer},
                       {DNNL_ARG_SRC_ITER, src_iter},
                       {DNNL_ARG_SRC_ITER_C, src_iter_c},
                       {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
                       {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
                       {DNNL_ARG_BIAS, expected_bias},
                       {DNNL_ARG_DST_LAYER, dst_layer},
                       {DNNL_ARG_DST_ITER, dst_iter},
                       {DNNL_ARG_DST_ITER_C, dst_iter_c},
                       {DNNL_ARG_DIFF_SRC_LAYER, diff_src_layer},
                       {DNNL_ARG_DIFF_SRC_ITER, diff_src_iter},
                       {DNNL_ARG_DIFF_SRC_ITER_C, diff_src_iter_c},
                       {DNNL_ARG_DIFF_WEIGHTS_LAYER, expected_diff_weights_layer},
                       {DNNL_ARG_DIFF_WEIGHTS_ITER, expected_diff_weights_iter},
                       {DNNL_ARG_DIFF_BIAS, expected_diff_bias},
                       {DNNL_ARG_DIFF_DST_LAYER, diff_dst_layer},
                       {DNNL_ARG_DIFF_DST_ITER, diff_dst_iter},
                       {DNNL_ARG_DIFF_DST_ITER_C, diff_dst_iter_c},
                       {DNNL_ARG_WORKSPACE, workspace}});

    expected_diff_weights_layer.reorder_to(diff_weights_layer);
    expected_diff_weights_iter.reorder_to(diff_weights_iter);
    expected_diff_bias.reorder_to(diff_bias);
  }
};

}  // namespace ideep

#endif