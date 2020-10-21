var webnn = webnn || {};
const nn = navigator.ml.getNeuralNetworkContext();

webnn.compile = async (graph) => {
  const operands = new Map();
  const builder = nn.createModelBuilder();
  console.log(graph);
  for (const input of graph.inputs) {
    console.log(input);
    for (const argument of input.arguments) {
      const type = argument.type.dataType;
      const dimensions = argument.type.shape.dimensions.map(d => d.toInteger());
      let desc = {type, dimensions};
      let operand = builder.input(argument.name, desc);
      console.log(operand);
      operands.set(argument.name, operand);
    }
  }

  for (const node of graph.nodes) {
    if (graph instanceof onnx.Graph) {
      buildOnnxNode(node, operands, builder);
    } else {
      throw new Error(`${graph} is not supported.`);
    }
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

function buildOnnxNode(node, operands, builder) {
  function toIntegerArray(array) {
    return array.map(v => v.toInteger());
  }

  function createDesc(tensorType) {
    const type = tensorType.dataType;
    const dimensions = toIntegerArray(tensorType.shape.dimensions);
    return {type, dimensions};
  }

  function createArrayBufferView(initializer) {
    if (initializer.type.dataType === 'float32') {
      return new Float32Array(initializer._data);
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
      if (a._initializer === null) {
        if (!operands.has(a.name))
          throw new Error(`Cannot find ${name} in operands.`);
        return operands.get(a.name);
      } else {
        const desc = createDesc(a.type);
        const data = createArrayBufferView(a._initializer);
        return builder.constant(desc, data);
      }
    }
  }

  function getValueForInput(name) {
    for (const p of node.inputs) {
      if (p.name === name) {
        const a = p.arguments[0];
        if (a._initializer === null) {
          throw new Error(`${name} does no have initializer.`);
        } else {
          return a._initializer._data;
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

  function getAttributeByName(name) {
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

  let handlers = {
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
      const shape = toIntegerArray(getValueForInput('shape'));
      const reshaped = builder.reshape(data, shape);
      addOperandForOutput(reshaped);
      console.log(reshaped);
    },

    Conv: () => {
      console.log('handle ONNX.Conv');
      const x = getOrCreateOperandForInput('X');
      const w = getOrCreateOperandForInput('W');
      const b = getOrCreateOperandForInput('B');;
      const kernel_shape = toIntegerArray(getAttributeByName('kernel_shape'));
      const strides = toIntegerArray(getAttributeByName('strides'));
      const dilations = toIntegerArray(getAttributeByName('dilations'));
      const groups = getAttributeByName('group').toInteger();
      const auto_pad = getAttributeByName('auto_pad');
      let padding;
      if (auto_pad === 'VALID') {
        padding = [0, 0, 0, 0];
      } else if (auto_pad === 'SAME_UPPER' || auto_pad === 'SAME_LOWER') {
        const shape_in = toIntegerArray(getInputByName('X').type.shape.dimensions);
        const pad_x = computePad(shape_in[2], strides[0], kernel_shape[0], auto_pad);
        const pad_y = computePad(shape_in[3], strides[1], kernel_shape[1], auto_pad);
        padding = [pad_x.head, pad_x.tail, pad_y.head, pad_y.tail];
      } else if (auto_pad.value === 'NOTSET') {
        padding = toIntegerArray(getAttributeByName('pads'));
      }
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
      const kernel_shape = toIntegerArray(getAttributeByName('kernel_shape'));
      const strides = toIntegerArray(getAttributeByName('strides'));
      const padding = toIntegerArray(getAttributeByName('pads'));
      const y = builder.maxPool2d(x, kernel_shape, padding, strides);
      addOperandForOutput(y);
      console.log(y);
    }
  };

  console.log(node);
  handlers[node.type]();
}


if (typeof module !== 'undefined' && typeof module.exports === 'object') {
  module.exports.compile = webnn.compile;
}