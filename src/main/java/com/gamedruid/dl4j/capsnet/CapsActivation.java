package com.gamedruid.dl4j.capsnet;

import org.nd4j.linalg.activations.BaseActivationFunction;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
//import org.nd4j.linalg.api.ops.impl.transforms.TanhDerivative;
//import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

//added by JFA
import org.nd4j.linalg.api.ops.impl.transforms.Sqrt;
import java.util.Arrays;


// testing

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
// import org.nd4j.linalg.api.ops.CustomOp;
// import org.nd4j.linalg.api.ops.DynamicCustomOp;
// import org.nd4j.linalg.api.ops.DynamicCustomOp;


// Capsule Activation Function
//
// Applies the Squash function to whatever
//

public class CapsActivation extends BaseActivationFunction
{

//=======================================================================//

 @Override
 public INDArray getActivation(INDArray in, boolean training)
 {
//def squash(vectors, axis=-1):
// """
// The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
// :param vectors: some vectors to be squashed, N-dim tensor
// :param axis: the axis to squash
// :return: a Tensor with same shape as input vectors
// """
//
// s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
// scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
// return scale * vectors

  System.out.println("@primcaps.getActivation()");


  // apply squash function
  System.out.println("  in = " + Arrays.toString(in.shape()));
  INDArray out = squash(in);
  System.out.println("  out = " + Arrays.toString(in.shape()));

  // scale the input vectors
  in.muli(0);
  in.addi(out);

  System.out.println("  result = " + Arrays.toString(in.shape()));

  return in;
 }

//=======================================================================//

 @Override
 public Pair<INDArray,INDArray> backprop(INDArray in, INDArray epsilon)
 {

  System.out.println("@primcaps.backprop()");

  // apply squash function
  System.out.println("  in = " + Arrays.toString(in.shape()));
  INDArray out = squash(in);
  System.out.println("  out = " + Arrays.toString(in.shape()));

  // Multiply with epsilon
  out.muli(epsilon);
  return new Pair<>(out, null);
 }

//=======================================================================//

// s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
// either:
//  scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
//  return scale * vectors
// or:
//  scale = K.sqrt(s_squared_norm + K.epsilon())
//  return vectors/scale

 INDArray squash(INDArray in)
 {
  System.out.println("@squash1()");

  //=====================//

  INDArray inb4 = in.dup();
  long[] i_shape = inb4.shape();
//  System.out.println("  sq1.inb4 = " + Arrays.toString( inb4.shape() ));
//  System.out.println("  sq1.inb4 = " + inb4 );

  INDArray s_temp = inb4.mul(inb4);
//  System.out.println("  sq1.s_temp = " + Arrays.toString( s_temp.shape() ));
//  System.out.println("  sq1.s_temp = " + s_temp );

// FIXME: this is where we lose a dimension
//  INDArray s_squared_norm = s_temp.sum(-1).reshape(100,256,16,1); // plugged for now
//  INDArray s_squared_norm = s_temp.sum(-1);
  INDArray s_squared_norm = s_temp.sum(s_temp.rank()-1);
//  System.out.println("  sq1.ssn1 = " + Arrays.toString( s_squared_norm.shape() ));
//  System.out.println("  sq1.ssn1 = " +  s_squared_norm );

  //=====================//
//  System.out.println("  sq1.ssn1 = " + Arrays.toString( inb4.toFloatVector() ));

  // prep scale
  double fuzz = 0.00000001;
  INDArray scale = Nd4j.getExecutioner().execAndReturn(new Sqrt(s_squared_norm.add(fuzz)));
//  System.out.println("  sq1.scale1 = " + Arrays.toString(scale.shape()));
//  System.out.println("  sq1.scale1 = " + scale );

 // FIXME: this is where it starts going weird
  scale = scale.reshape(i_shape[0],i_shape[1],i_shape[2],1);
//  System.out.println("  sq1.scale2 = " + Arrays.toString( scale.shape() ));
//  System.out.println("  sq1.scale2 = " + scale );

 // FIXME: this is where it really blows up
  scale = scale.repeat(3,i_shape[3]); // FIXME: plugged for now
//  System.out.println("  sq1.scale3 = " + Arrays.toString( scale.shape() ));
//  System.out.println("  sq1.scale3 = " + scale );

  //=====================//

  // scale the input vectors
  INDArray out = in.div(scale);
//  System.out.println("  sq1.out = " + Arrays.toString( out.shape() ));
//  System.out.println("  sq1.out = " + out );

  return out;
 }

//=======================================================================//

}
