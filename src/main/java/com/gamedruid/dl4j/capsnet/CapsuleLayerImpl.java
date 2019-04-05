package com.gamedruid.dl4j.capsnet;

import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.activations.IActivation;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

//added by JFA
import org.nd4j.linalg.api.ops.impl.accum.Dot;
import org.nd4j.linalg.api.ops.impl.transforms.Sqrt;
import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import org.nd4j.linalg.api.ops.impl.accum.Mmul;
import org.nd4j.linalg.api.blas.params.MMulTranspose;


/**
 * Layer (implementation) class for DigitCaps layer (... plus batch_dot, squash function, etc)
 *
 * @author JF Aubertin
 */
public class CapsuleLayerImpl extends BaseLayer<CapsuleLayer>
{

 public CapsuleLayerImpl(NeuralNetConfiguration conf)
 {
  super(conf);
 }

//=======================================================================//
//=======================================================================//
//=======================================================================//

 @Override
 public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr)
 {
  //System.out.println("@caps.activate()");

  // int miniBatch = (int) input.size(0);
  // int inH = (int) input.size(2);
  // int inW = (int) input.size(3);

  //=======================================================================//

//  int columns = output.columns();

  INDArray inputs = input.dup();
  long[] i_shape = inputs.shape();

  // FIXME: should come from inputs
  int i_num_capsules = ((CapsuleLayer) conf.getLayer()).getInputNumCapsules(); // was 10 ... 7
  int i_dim_capsules = ((CapsuleLayer) conf.getLayer()).getInputDimCapsules(); // was 16 ... 11

//  int i_num_capsules=(int) i_shape[1]; // was  8?  16?
//  int i_dim_capsules=(int) i_shape[2]; // was 16? 256?

  inputs = inputs.reshape(new long[]{i_shape[0],i_num_capsules,i_dim_capsules});

  //=======================================================================//

  INDArray outputs = preOutput(training, workspaceMgr);
  long[] o_shape = outputs.shape();
    System.out.println("  a.outputs = " + Arrays.toString(outputs.shape()));

  int num_capsules = ((CapsuleLayer) conf.getLayer()).getOutputNumCapsules(); // was 10 ... 7
  int dim_capsules = ((CapsuleLayer) conf.getLayer()).getOutputDimCapsules(); // was 16 ... 11

  outputs = outputs.reshape(new long[]{o_shape[0],num_capsules,dim_capsules});
  System.out.println("  a.outputs = " + Arrays.toString(outputs.shape()));

  //=======================================================================//

  INDArray weights = getParamWithNoise(DefaultParamInitializer.WEIGHT_KEY, true, workspaceMgr);

  // FIXME: weights should already be the right shape
  // weights shape = output_num_caps, input_num_caps, output_dim_caps, input_dim_caps
  weights = weights.reshape(i_num_capsules, i_dim_capsules, num_capsules, dim_capsules);
  weights = weights.permute(2,0,3,1);
  //=======================================================================//

  // this is where the magic happens...
  INDArray ret = dynamicRouting(inputs, weights);

  //=======================================================================//

  //IActivation function instances modify the activation functions in-place
  IActivation activation1 = layerConf().getActivationFn();
  activation1.getActivation(ret, training);

  return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, ret);
 }

//=======================================================================//
//=======================================================================//
//=======================================================================//

 @Override
 public boolean isPretrainLayer()
 {
  return false;
 }

//=======================================================================//
//=======================================================================//
//=======================================================================//

 @Override
 public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr)
 {
  // System.out.println("caps.backpropGradient()");
  assertInputSet(true);

  //=======================================================================//

  INDArray inputs = input.dup();
  long[] i_shape = inputs.shape();

  // FIXME: should come from inputs
  int i_num_capsules = ((CapsuleLayer) conf.getLayer()).getInputNumCapsules(); // was 16  = PrimaryCaps "depth"
  int i_dim_capsules = ((CapsuleLayer) conf.getLayer()).getInputDimCapsules(); // was 256 = PrimaryCaps width * height

//  int i_num_capsules=(int) i_shape[1]; // was  8?  16?
//  int i_dim_capsules=(int) i_shape[2]; // was 16? 256?

  inputs = inputs.reshape(new long[]{i_shape[0],i_num_capsules,i_dim_capsules});

  //=======================================================================//

  INDArray outputs = preOutput(true, workspaceMgr);
  long[] o_shape = outputs.shape();

  int num_capsules = ((CapsuleLayer) conf.getLayer()).getOutputNumCapsules(); // was 10 ... 7
  int dim_capsules = ((CapsuleLayer) conf.getLayer()).getOutputDimCapsules(); // was 16 ... 11

  outputs = outputs.reshape(new long[]{i_shape[0],num_capsules,dim_capsules});

  //=======================================================================//

  INDArray weights = getParamWithNoise(DefaultParamInitializer.WEIGHT_KEY, true, workspaceMgr);
  INDArray bias = getParamWithNoise(DefaultParamInitializer.BIAS_KEY, true, workspaceMgr);

  // FIXME: weights should already be the right shape
  // weights shape = output_num_caps, input_num_caps, output_dim_caps, input_dim_caps
  weights = weights.reshape(i_num_capsules, i_dim_capsules, num_capsules, dim_capsules);
  weights = weights.permute(2,0,3,1);

  //=======================================================================//

  // this is where the magic happens...
  INDArray activationDerivative = dynamicRouting(inputs, weights);


  IActivation activation1 = layerConf().getActivationFn();
  activation1.backprop(activationDerivative, epsilon);

  //=======================================================================//

  //The remaining code for this method: just copy & pasted from BaseLayer.backpropGradient
  //  INDArray delta = epsilon.muli(activationDerivative);
  if (maskArray != null)   { activationDerivative.muliColumnVector(maskArray); }

  Gradient ret = new DefaultGradient();

  INDArray weightGrad = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY); //f order
  Nd4j.gemm(input, activationDerivative, weightGrad, true, false, 1.0, 0.0);
  INDArray biasGrad = gradientViews.get(DefaultParamInitializer.BIAS_KEY);
  biasGrad.assign(activationDerivative.sum(0));  //TODO: do this without the assign

  ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGrad);
  ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, biasGrad);

  INDArray epsilonNext = params.get(DefaultParamInitializer.WEIGHT_KEY).mmul(activationDerivative.transpose()).transpose();

  return new Pair<>(ret, workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilonNext));
 }

//=======================================================================//
//=======================================================================//
//=======================================================================//

//=======================================================================//

 protected INDArray dynamicRouting(INDArray inputs, INDArray weights)
 {
  // System.out.println("\ndynamicRouting()\n");

  // FIXME: assert self.routings > 0, 'The routings should be > 0.'

  //=======================================================================//

  int num_capsules = ((CapsuleLayer) conf.getLayer()).getOutputNumCapsules();
  int dim_capsules = ((CapsuleLayer) conf.getLayer()).getOutputDimCapsules();
  int routings     = ((CapsuleLayer) conf.getLayer()).getRoutings();

  //=======================================================================//

  long[] i_shape = inputs.shape();  // inputs.shape   (batch_size, input_num_caps, input_dim_caps)

  int batch_size         = (int) i_shape[0];
  int input_num_capsules = (int) i_shape[1];
  int input_dim_capsules = (int) i_shape[2];

  //=======================================================================//

  INDArray inputs_expand = inputs.reshape(i_shape[0], 1, i_shape[1],i_shape[2]); // add dimension at 2

  // FIXME: this part is slower for some reason?
  // NOTE: this seems to multiply dim[1] of inputs?  ... inputs x num_capsule?
  System.out.println("tiling inputs_tiled with inputs_expand...");
  INDArray inputs_tiled = Nd4j.tile(inputs_expand, new int[]{1, num_capsules, 1, 1} );

  //=====================//
  INDArray inputs_hat = Nd4j.zeros(new long[]{batch_size, num_capsules, input_num_capsules, dim_capsules}); // FIXME: shape?

  // inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)
  int len=(int) inputs_tiled.shape()[0];
  for (int i=0; i<len; i++)
  {
   //System.out.println("=== inputs_tiled ["+i+"] ===");
   INDArray x = inputs_tiled.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());

   INDArray z = bd_23( x, weights);  // FIXME: should be generic if possible = batch_dot( x, weights, new int[]{2,3} );  // axes
   inputs_hat.putRow(i, z);          // FIXME: is there a better/faster way?
  }

  //=====================//

  long[] ih_shape = inputs_hat.shape();
  INDArray b = Nd4j.zeros(ih_shape[0], num_capsules, input_num_capsules); // shape
  INDArray outputs = Nd4j.zeros(1, num_capsules, dim_capsules); // FIXME: create in appropriate workspace for better performance?

  for (int i=0; i<routings; i++)
  {
   //System.out.println("=== ROUTING["+i+"] ===");
   INDArray c = softmax1(b, 1);
   outputs = squash2(bd_cihat_22(c, inputs_hat));

   // skip for the last routing
   if (i < (routings-1))   b.addi( bd_23(outputs, inputs_hat) );
  }

  outputs=outputs.reshape(new long[]{batch_size, num_capsules*dim_capsules});

  return outputs;
 }

//=======================================================================//
//=======================================================================//
//=======================================================================//

 static private INDArray softmax1(INDArray in, int axis)
 {
  INDArray out = in.dup();
  CustomOp op = DynamicCustomOp.builder("softmax").addInputs(in).addOutputs(out).addIntegerArguments(axis).build();
  Nd4j.getExecutioner().exec( op );
  return out;
 }

//=======================================================================//

// s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
// either:
//  scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
//  return scale * vectors
// or:
//  scale = K.sqrt(s_squared_norm + K.epsilon())
//  return vectors/scale

 static private INDArray squash2(INDArray in)
 {
//  System.out.println("@squash2()");

  //=====================//

  INDArray inb4 = in.dup();
  long[] i_shape = inb4.shape();
  INDArray s_temp = inb4.mul(inb4);
  INDArray s_squared_norm = s_temp.sum(s_temp.rank()-1);

  // get the scale
  double fuzz = 0.00000001;
  INDArray scale = Nd4j.getExecutioner().execAndReturn(new Sqrt(s_squared_norm.add(fuzz)));

  // scale the input vectors
  scale = scale.reshape(i_shape[0],i_shape[1],1); // FIXME: plugged for now
  scale = scale.repeat(2,i_shape[2]); // FIXME: plugged for now

  INDArray out = in.div(scale);

  return out;
 }

//=======================================================================//

 // insert dimension of size "1" at shape[dim]
 static private INDArray expand_dims(INDArray orig, int dim)
 {
  if (dim < 0)  dim = orig.rank() + dim; // for python users

  INDArray arr = orig.dup();
  long[] lArr = arr.shape();

  // int rank = arr.rank();
  long[] shape = new long[arr.rank()+1];
  for (int i=0; i<shape.length; i++)
  {
   if (i==dim)      shape[i]=1;
   else if (i>dim)  shape[i]=lArr[i-1];
   else             shape[i]=lArr[i];
  }

  return arr.reshape(shape);
 }

//=======================================================================//


 // batch_dot with x.shape(3 dims), y.shape(4 dims), axes = 2,3
 static private INDArray bd_23(INDArray x, INDArray y)
 {
  int a0 = 2; // axes[0];
  int a1 = 3; // axes[1];

  long[] x_shape = x.shape();
  long[] y_shape = y.shape();

//  System.out.println("  bd23.x = " + Arrays.toString(x.shape()));
//  System.out.println("  bd23.y = " + Arrays.toString(y.shape()));

  INDArray result = Nd4j.zeros(new long[]{ x_shape[0], x_shape[1], 1, y_shape[2] });

  x = expand_dims(x, 2);
  y = y.swapAxes(3,2);

  //System.out.println("  bd23.x = " + Arrays.toString(x.shape()));
  //System.out.println("  bd23.y = " + Arrays.toString(y.shape()));

  Nd4j.getExecutioner().exec(new Mmul(x,y,result, MMulTranspose.allFalse()));
  //System.out.println("  bd23.result = " + Arrays.toString(result.shape()));

  long[] r_shape = result.shape();

  return result.reshape(new long[]{ r_shape[0], r_shape[1], r_shape[3] });
 }

//=======================================================================//

 // batch_dot with x.shape(3 dims), y.shape(4 dims), axes = 2,2
 static private INDArray bd_cihat_22(INDArray x, INDArray y)
 {
  int a0 = 2; // axes[0];
  int a1 = 2; // axes[1];

  long[] x_shape = x.shape();
  long[] y_shape = y.shape();

//  System.out.println("  bd22.x = " + Arrays.toString(x.shape()));
//  System.out.println("  bd22.y = " + Arrays.toString(y.shape()));

  INDArray result = Nd4j.zeros(new long[]{ x_shape[0], x_shape[1], 1, y_shape[3] });

  x = expand_dims(x, 2);
  //y = y.swapAxes(3,2);

//  System.out.println("  bd22.x = " + Arrays.toString(x.shape()));
//  System.out.println("  bd22.y = " + Arrays.toString(y.shape()));

  Nd4j.getExecutioner().exec(new Mmul(x,y,result, MMulTranspose.allFalse()));
//  System.out.println("  bd22.result = " + Arrays.toString(result.shape()));

  long[] r_shape = result.shape();

  return result.reshape(new long[]{ r_shape[0], r_shape[1], r_shape[3] });
 }

//=======================================================================//
//=======================================================================//
//=======================================================================//

}
