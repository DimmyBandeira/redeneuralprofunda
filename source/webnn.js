var webnn = webnn || {};
const nn = navigator.ml.getNeuralNetworkContext();

function toIntegerArray(array) {
  return array.map(v => v instanceof base.Int64 ? v.toInteger() : v);
}

function createDesc(tensorType) {
  const type = tensorType.dataType;
  const dimensions = toIntegerArray(tensorType.shape.dimensions);
  return {type, dimensions};
}

webnn.compile = async (graph) => {
  const operands = new Map();
  const builder = nn.createModelBuilder();
  console.log(graph);
  for (const input of graph.inputs) {
    console.log(input);
    for (const argument of input.arguments) {
      const desc = createDesc(argument.type);
      const operand = builder.input(argument.name, desc);
      console.log(operand);
      operands.set(argument.name, operand);
    }
  }

  for (const node of graph.nodes) {
    buildNode(node, operands, builder);
  }

  let outputs = {};
  for (const output of graph.outputs) {
    console.log(output);
    for (const argument of output.arguments) {
      outputs[argument.name] = operands.get(argument.name);
    }
  }
  const model = builder.createModel(outputs);
  return await model.compile();
}

function buildNode(node, operands, builder) {
  function createArrayBufferView(initializer) {
    if (initializer.type.dataType === 'float32') {
      return new Float32Array(initializer.value.flat(initializer.type.shape.dimensions.length));
    } else {
      throw new Error(`${dataType} is not supported.`);
    }
  }

  function getInputByName(name) {
    for (const p of node.inputs) {
      if (p.name === name) {
        return p.arguments[0];
      }
    }
    // It may be optional paramter.
    return undefined;
  }

  function getOrCreateOperandForInput(name) {
    const a = getInputByName(name);
    if (a === undefined) {
      return undefined;
    } else {
      if (a.initializer === null) {
        if (!operands.has(a.name))
          throw new Error(`Cannot find ${name} in operands.`);
        return operands.get(a.name);
      } else {
        const desc = createDesc(a.type);
        const data = createArrayBufferView(a.initializer);
        return builder.constant(desc, data);
      }
    }
  }

  function getValueOfInput(name) {
    for (const p of node.inputs) {
      if (p.name === name) {
        const a = p.arguments[0];
        if (a.initializer === null) {
          throw new Error(`${name} does no have initializer.`);
        } else {
          return a.initializer.value;
        }
      }
    }
    throw new Error(`Cannot find ${name} in parameters.`);
  }

  function addOperandForOutput(operand) {
    const name = node.outputs[0].arguments[0].name;
    if (operands.has(name))
      throw new Error(`${name} is duplicate.`)
    operands.set(name, operand);
  }

  function getValueOfAttribute(name) {
    for (const a of node.attributes) {
      if (a.name === name) {
        return a.value;
      }
    }
    return undefined;
  }

  function computePad(in_dim, stride, kernel, pad_type) {
    const legacy_target_size = (in_dim + stride - 1) / stride;
    const pad_needed = (legacy_target_size - 1) * stride + kernel - in_dim;
  
    let head, tail;
    if (pad_type === 'SAME_LOWER') {
      head = (pad_needed + 1) / 2;
    } else {
      head = pad_needed / 2;
    }
  
    tail = pad_needed - head;
    return {head, tail};
  }

  function buildTfliteNode() {
    function buildFusedActivation(output, fused_activation_function) {
      if (fused_activation_function === 'NONE') {
        // do nothing.
      } else if (fused_activation_function === 'RELU') {
        output = builder.relu(output);
        console.log(output);
      } else {
        throw new Error(`${fused_activation_function} is not supported.`);
      }
      return output;
    }

    function buildPadding(input_shape, kernel_shape, strides, auto_pad) {
      let padding;
      if (auto_pad === 'VALID') {
        padding = [0, 0, 0, 0];
      } else if (auto_pad === 'SAME') {
        const pad_x = computePad(input_shape[1], strides[0], kernel_shape[1], auto_pad);
        const pad_y = computePad(input_shape[2], strides[1], kernel_shape[2], auto_pad);
        padding = [pad_x.head, pad_x.tail, pad_y.head, pad_y.tail];
      }
      return padding;
    }

    const handlers = {
      Reshape: () => {
        console.log('handle TFLite.Reshape');
        const data = getOrCreateOperandForInput('data');
        const shape = getValueOfInput('shape');
        const reshaped = builder.reshape(data, shape);
        addOperandForOutput(reshaped);
        console.log(reshaped);
      },
      Conv2D: () => {
        console.log('handle TFLite.Conv2D');
        const input = getOrCreateOperandForInput('input');
        let filter = getOrCreateOperandForInput('filter');
        const bias = getOrCreateOperandForInput('bias');
        const stride_h = getValueOfAttribute('stride_h');
        const stride_w = getValueOfAttribute('stride_w');
        const strides = [stride_h, stride_w];
        const dilation_h_factor = getValueOfAttribute('dilation_h_factor');
        const dilation_w_factor = getValueOfAttribute('dilation_w_factor');
        const dilations = [dilation_h_factor, dilation_w_factor];
        const fused_activation_function = getValueOfAttribute('fused_activation_function');
        const input_shape = toIntegerArray(getInputByName('input').type.shape.dimensions);
        const kernel_shape = toIntegerArray(getInputByName('filter').type.shape.dimensions);
        const auto_pad = getValueOfAttribute('padding');
        const padding = buildPadding(input_shape, kernel_shape, strides, auto_pad);
        // TFLite filter: [depth_out, filter_height, filter_width, depth_in]
        // webnn nhwc filter: [height, width, input_channels/groups, output_channels]
        filter = builder.transpose(filter, [1, 2, 3, 0]);
        let output = builder.conv2d(input, filter, padding, strides, dilations, 1, 'nhwc');
        console.log(output);
        if (bias !== undefined) {
          output = builder.add(output, bias);
          console.log(output);
        }
        output = buildFusedActivation(output, fused_activation_function);
        addOperandForOutput(output);
      },
      MaxPool2D: () => {
        console.log('handle TFLite.MaxPool2D');
        const input = getOrCreateOperandForInput('input');
        const filter_height = getValueOfAttribute('filter_height');
        const filter_width = getValueOfAttribute('filter_width');
        const windowDimensions = [filter_height, filter_width];
        const stride_h = getValueOfAttribute('stride_h');
        const stride_w = getValueOfAttribute('stride_w');
        const strides = [stride_h, stride_w];
        const fused_activation_function = getValueOfAttribute('fused_activation_function');
        const auto_pad = getValueOfAttribute('padding');
        const input_shape = toIntegerArray(getInputByName('input').type.shape.dimensions);
        const padding = buildPadding(input_shape, windowDimensions, strides, auto_pad);
        let output = builder.maxPool2d(input, windowDimensions, padding, strides, [1, 1], 'nhwc');
        console.log(output);
        output = buildFusedActivation(output, fused_activation_function);
        addOperandForOutput(output);
      },
      FullyConnected: () => {
        console.log('handle TFLite.FullyConnected');
        const input = getOrCreateOperandForInput('input');
        const weights = getOrCreateOperandForInput('weights');
        const bias = getOrCreateOperandForInput('bias');
        const fused_activation_function = getValueOfAttribute('fused_activation_function');
        let output = builder.matmul(input, builder.transpose(weights));
        console.log(output);
        output = buildFusedActivation(output, fused_activation_function);
        addOperandForOutput(output);
      }
    };
    if (node.type in handlers) {
      handlers[node.type]();
    } else {
      throw new Error(`TFLite ${node.type} is not supported.`);
    }
  }

  function buildOnnxNode() {
    function buildPadding(input_shape, kernel_shape, strides, auto_pad) {
      let padding;
      if (auto_pad === 'VALID') {
        padding = [0, 0, 0, 0];
      } else if (auto_pad === 'SAME_UPPER' || auto_pad === 'SAME_LOWER') {
        const pad_x = computePad(input_shape[2], strides[0], kernel_shape[0], auto_pad);
        const pad_y = computePad(input_shape[3], strides[1], kernel_shape[1], auto_pad);
        padding = [pad_x.head, pad_x.tail, pad_y.head, pad_y.tail];
      } else if (auto_pad === 'NOTSET') {
        padding = toIntegerArray(getValueOfAttribute('pads'));
      }
      return padding;
    }
    const handlers = {
      MatMul: () => {
        console.log('handle ONNX.MatMul');
        const a = getOrCreateOperandForInput('A');
        const b = getOrCreateOperandForInput('B');
        const c = builder.matmul(a, b);
        addOperandForOutput(c);
        console.log(c);
      },
      Add: () => {
        console.log('handle ONNX.Add');
        const a = getOrCreateOperandForInput('A');
        const b = getOrCreateOperandForInput('B');
        const c = builder.add(a, b);
        addOperandForOutput(c);
        console.log(c);
      },
      Reshape: () => {
        console.log('handle ONNX.Reshape');
        const data = getOrCreateOperandForInput('data');
        const shape = toIntegerArray(getValueOfInput('shape'));
        const reshaped = builder.reshape(data, shape);
        addOperandForOutput(reshaped);
        console.log(reshaped);
      },
      Conv: () => {
        console.log('handle ONNX.Conv');
        const x = getOrCreateOperandForInput('X');
        const x_shape = toIntegerArray(getInputByName('X').type.shape.dimensions);
        const w = getOrCreateOperandForInput('W');
        const b = getOrCreateOperandForInput('B');;
        const kernel_shape = toIntegerArray(getValueOfAttribute('kernel_shape'));
        const strides = toIntegerArray(getValueOfAttribute('strides'));
        const dilations = toIntegerArray(getValueOfAttribute('dilations'));
        const groups = getValueOfAttribute('group').toInteger();
        const auto_pad = getValueOfAttribute('auto_pad');
        const padding = buildPadding(x_shape, kernel_shape, strides, auto_pad);
        let y = builder.conv2d(x, w, padding, strides, dilations, groups);
        if (b !== undefined) y = builder.add(y, b);
        addOperandForOutput(y);
        console.log(y);
      },
      Relu: () => {
        console.log('handle ONNX.Relu');
        const x = getOrCreateOperandForInput('X');
        const y = builder.relu(x);
        addOperandForOutput(y);
        console.log(y);
      },
      MaxPool: () => {
        console.log('handle ONNX.MaxPool');
        const x = getOrCreateOperandForInput('X');
        const x_shape = toIntegerArray(getInputByName('X').type.shape.dimensions);
        const kernel_shape = toIntegerArray(getValueOfAttribute('kernel_shape'));
        const strides = toIntegerArray(getValueOfAttribute('strides'));
        const auto_pad = getValueOfAttribute('auto_pad');
        const padding = buildPadding(x_shape, kernel_shape, strides, auto_pad);
        const y = builder.maxPool2d(x, kernel_shape, padding, strides);
        addOperandForOutput(y);
        console.log(y);
      }
    };
    if (node.type in handlers) {
      handlers[node.type]();
    } else {
      throw new Error(`ONNX ${node.type} is not supported.`);
    }
  }

  console.log(node);
  if (typeof onnx !== 'undefined' && node instanceof onnx.Node) {
    buildOnnxNode();
  } else if (typeof tflite !== 'undefined' && node instanceof tflite.Node) {
    buildTfliteNode();
  } else {
    throw new Error('Format is not supported.');
  }
}

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
  module.exports.compile = webnn.compile;
}
