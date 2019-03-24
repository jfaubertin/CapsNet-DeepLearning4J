package com.gamedruid.dl4j.capsnet;

//import org.deeplearning4j.examples.misc.activationfunctions.CapsActivation;
//import org.deeplearning4j.examples.misc.customlayers.layer.CapsuleLayer;


import org.datavec.image.loader.CifarLoader;
//import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.nd4j.evaluation.classification.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;

import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.api.storage.StatsStorage;

import org.deeplearning4j.util.ModelSerializer;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
//import org.nd4j.util.StringUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//import javax.swing.*;
//import javax.swing.border.LineBorder;
//import javax.swing.border.TitledBorder;
//import java.awt.*;
//import java.awt.event.ActionEvent;
//import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
//import java.util.Arrays;



/**
 * train Capsule model on Cifar
 *
 * @author JF Aubertin
 */

public class Cifar
{
//=======================================================================//

 private static final File DATA_PATH = new File(System.getProperty("user.dir"));
 protected static final Logger log = LoggerFactory.getLogger(Cifar.class);

 private static int height = 32;
 private static int width = 32;
 private static int channels = 3;

 private static int numLabels = CifarLoader.NUM_LABELS;
 private static int numSamples = 50000;
 private static int batchSize = 100;

 private static int freqIterations = 50;
 private static int seed = 123;
 private static boolean preProcessCifar = false;//use Zagoruyko's preprocess for Cifar
 private static int epochs = 50;

//=======================================================================//

 public static void main(String[] args) throws Exception
 {
  Nd4j.setDataType(DataBuffer.Type.FLOAT);
  Cifar cf = new Cifar();

  //train model and eval model
  MultiLayerNetwork model = cf.createModel();

  UIServer uiServer = UIServer.getInstance();
  StatsStorage statsStorage = new InMemoryStatsStorage();
  uiServer.attach(statsStorage);
  model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(freqIterations));

  CifarDataSetIterator cifar     = new CifarDataSetIterator(batchSize, numSamples, new int[] {height, width, channels}, preProcessCifar, true);
  CifarDataSetIterator cifarEval = new CifarDataSetIterator(batchSize, 10000,      new int[] {height, width, channels}, preProcessCifar, false);

  for (int i=0; i<epochs; i++)
  {
   System.out.println("Epoch====================" + i);
   model.fit(cifar);
   cf.saveModel(model, "trainCapsulesOnCifar.json");
  }

  log.info("=====eval model==========");
  Evaluation eval = new Evaluation(cifarEval.getLabels());
  while(cifarEval.hasNext())
  {
   DataSet testDS = cifarEval.next(batchSize);
   INDArray output = model.output(testDS.getFeatures());
   eval.eval(testDS.getLabels(), output);
  }
  System.out.println(eval.stats());

 }

//=======================================================================//


 public MultiLayerNetwork createModel()
   throws IOException
 {
  log.info("Creating model...");
  MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
   .seed(seed)
   .cacheMode(CacheMode.DEVICE)
   .updater(new Adam(1e-2))
   .biasUpdater(new Adam(1e-2*2))
   .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
   .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
   .l1(1e-4)
   .l2(5 * 1e-4)
   .list()

//=======================================================================//

   // kernelSize, stride, padding
   .layer(0, new ConvolutionLayer.Builder(new int[]{4,4}, new int[]{2, 2}, new int[] {0,0}).name("cnn1").convolutionMode(ConvolutionMode.Same)
    .nIn(3).nOut(32).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)//.learningRateDecayPolicy(LearningRatePolicy.Step)
    .biasInit(1e-2).build())

   .layer(1, new ConvolutionLayer.Builder(new int[]{4,4}, new int[] {2,2}, new int[] {0,0}).name("cnn2").convolutionMode(ConvolutionMode.Same)
    .nOut(32).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
    .biasInit(1e-2).build())

   // total 16 x (16x16=256) = 4096
   .layer(2, new ConvolutionLayer.Builder(new int[]{4,4}, new int[] {2,2}, new int[] {0,0})
      .name("PrimaryCaps1.conv2d").convolutionMode(ConvolutionMode.Same)
    .nOut(32).weightInit(WeightInit.XAVIER_UNIFORM)
    .activation(new CapsActivation())
    .biasInit(1e-2).build())


   // InputNumCapsules = PrimaryCaps depth
   // InputDimCapsules = PrimaryCaps width * height
   .layer(3, new CapsuleLayer.Builder().name("DigitCaps1")
     .setInputNumCapsules(32).setInputDimCapsules(16) // FIXME: input should already have this shape
     .setOutputNumCapsules(16).setOutputDimCapsules(16)
     .setRoutings(8)
     .nOut(256) // FIXME: nOut = OutputNumCapsules * OutputDimCapsules
     .updater(new Adam(1e-3)).biasInit(1e-3).biasUpdater(new Adam(1e-3*2)).build())


   .layer(4+0, new DenseLayer.Builder().name("ffn1").nOut(256).updater(new Adam(1e-3))
     .biasInit(1e-3).biasUpdater(new Adam(1e-3*2)).build())

   .layer(5+0,new DropoutLayer.Builder().name("dropout2").dropOut(0.2).build())
   .layer(6+0, new DenseLayer.Builder().name("ffn2").nOut(256).biasInit(1e-2).build())
   .layer(7+0,new DropoutLayer.Builder().name("dropout2").dropOut(0.2).build())
   .layer(8+0, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
    .name("output")
    .nOut(numLabels)
    .activation(Activation.SOFTMAX)
    .build())
//   .inputPreProcessor(3, new ReshapePreprocessor( new long[]{100,7,11}, new long[]{100,77} ) )

//=======================================================================//

   .backprop(true).pretrain(false)
   .setInputType(InputType.convolutional(height, width, channels))
   .build();

  MultiLayerNetwork model = new MultiLayerNetwork(conf);
  model.init();
  return model;
 }

//=======================================================================//

 public MultiLayerNetwork saveModel(MultiLayerNetwork model, String fileName)
 {
  File locationModelFile = new File(DATA_PATH, fileName);
  boolean saveUpdater = false;
  try
  {
   ModelSerializer.writeModel(model, locationModelFile, saveUpdater);
   log.info("trained model was saved to {}", locationModelFile);
  } catch (Exception e) { log.error("Failed to save model!",e); }
  return model;
 }

//=======================================================================//

}
