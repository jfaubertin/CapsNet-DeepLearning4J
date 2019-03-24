package com.gamedruid.dl4j.capsnet;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
//import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
//import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
//import org.nd4j.linalg.activations.Activation;
//import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

//=======================================================================//

/**
 * Layer configuration class for the Capsule Layer (aka DigitCaps)
 *
 * @author JF Aubertin
 */
public class CapsuleLayer extends FeedForwardLayer
{
//=======================================================================//

 private int i_num_capsules = 0; // 16 // i (int): Number of input (child?) capsules
 private int i_dim_capsules = 0; // 16 // m (int): Vector length of input (child?) capsules

 private int o_num_capsules = 0; // 10  // j (int): Number of output (parent?) capsules.
 private int o_dim_capsules = 0; // 16  // n (int): Vector length of output (parent?) capsules.

 private int batch_size     = 0; // 100  // b (int): batch // FIXME: should this come from input[0]?
 private int routings       = 0; // 8

 // REM:  input.shape = [ batch,  input_num_caps,  input_dim_caps ]
 // REM: output.shape = [ batch, output_num_caps, output_dim_caps ]


//=======================================================================//

 public CapsuleLayer()
 {
  // empty constructor
 }

 private CapsuleLayer(Builder builder)
 {
  super(builder);

  // TODO: infer num/dim of input capules from input size? or vice-versa?
  // TODO: infer num/dim of output capules from output size? or vice-versa?

  this.i_num_capsules = builder.i_num_capsules;
  this.i_dim_capsules = builder.i_dim_capsules;
  this.o_num_capsules = builder.o_num_capsules;
  this.o_dim_capsules = builder.o_dim_capsules;
  this.batch_size     = builder.batch_size;
  this.routings       = builder.routings;
 }

//=======================================================================//

 @Override
 public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> iterationListeners,
                          int layerIndex, INDArray layerParamsView, boolean initializeParams)
 {
  CapsuleLayerImpl myCustomLayer = new CapsuleLayerImpl(conf);
  myCustomLayer.setListeners(iterationListeners);
  myCustomLayer.setIndex(layerIndex);

  myCustomLayer.setParamsViewArray(layerParamsView);

  Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
  myCustomLayer.setParamTable(paramTable);
  myCustomLayer.setConf(conf);
  return myCustomLayer;
 }

 @Override
 public ParamInitializer initializer()
 {
  // TODO: may need to implement a custom parameter initializer?
  return DefaultParamInitializer.getInstance();
 }

 @Override
 public LayerMemoryReport getMemoryReport(InputType inputType)
 {
  InputType outputType = getOutputType(-1, inputType);

  long numParams = initializer().numParams(this);
  int updaterStateSize = (int)getIUpdater().stateSize(numParams);

  int trainSizeFixed = 0;
  int trainSizeVariable = 0;
  if(getIDropout() != null)   trainSizeVariable += inputType.arrayElementsPerExample();

  trainSizeVariable += outputType.arrayElementsPerExample();

  return new LayerMemoryReport.Builder(layerName, CapsuleLayer.class, inputType, outputType)
   .standardMemory(numParams, updaterStateSize)
   .workingMemory(0, 0, trainSizeFixed, trainSizeVariable)  //No additional memory (beyond activations) for inference
   .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
   .build();
 }

//=======================================================================//

  public int getInputNumCapsules()
  {
   return this.i_num_capsules;
  }

  public int getInputDimCapsules()
  {
   return this.i_dim_capsules;
  }

  public int getOutputNumCapsules()
  {
   return this.o_num_capsules;
  }

  public int getOutputDimCapsules()
  {
   return this.o_dim_capsules;
  }

  public int getRoutings()
  {
   return this.routings;
  }

//=======================================================================//
//=======================================================================//
//=======================================================================//

 public static class Builder extends FeedForwardLayer.Builder<Builder>
 {
  //=======================================================================//

  // Number and Vector Length of Input (child?) capsules
  private int i_num_capsules = 0; // 16
  private int i_dim_capsules = 0; // 16

  // Number and Vector Length of Output (parent?) capsules
  private int o_num_capsules = 0; // 10
  private int o_dim_capsules = 0; // 16

  private int batch_size     = 0; // 100  // FIXME: should this come from input[0]?
  private int routings       = 0; // 8


  public Builder setInputNumCapsules(int num)
  {
   this.i_num_capsules=num;
   return this;
  }

  public Builder setInputDimCapsules(int num)
  {
   this.i_dim_capsules=num;
   return this;
  }

  public Builder setOutputNumCapsules(int num)
  {
   this.o_num_capsules=num;
   return this;
  }

  public Builder setOutputDimCapsules(int num)
  {
   this.o_dim_capsules=num;
   return this;
  }

  public Builder setRoutings(int num)
  {
   this.routings=num;
   return this;
  }

  //=======================================================================//

  @Override
  @SuppressWarnings("unchecked")  //To stop warnings about unchecked cast. Not required.
  public CapsuleLayer build()
  {
   return new CapsuleLayer(this);
  }
 }

//=======================================================================//

}
