let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var rockSamples=0, paperSamples=0, scissorsSamples=0, spockSamples=0, lizardSamples=0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(5);
    
  // In the space below create a neural network that can classify hand gestures
  // corresponding to rock, paper, scissors, lizard, and spock. The first layer
  // of your network should be a flatten layer that takes as input the output
  // from the pre-trained MobileNet model. Since we have 5 classes, your output
  // layer should have 5 units and a softmax activation function. You are free
  // to use as many hidden layers and neurons as you like.  
  // HINT: Take a look at the Rock-Paper-Scissors example. We also suggest
  // using ReLu activation functions where applicable.
  model = tf.sequential({
    layers: [
      tf.layers.conv2d({inputShape: mobilenet.outputs[0].shape.slice(1), kernelSize: 1, filters: 128, activation: 'relu', kernel_initializer: 'he_uniform'}),
      tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu', kernel_initializer: 'he_uniform'}),
      tf.layers.maxPooling2d({poolSize: [2, 2]}),
      tf.layers.flatten(),
      // tf.layers.dense({units: 32, activation: "relu"}),
      tf.layers.dense({units: 5, activation: "softmax"})
      // YOUR CODE HERE
    ]
  });
  
  // Set the optimizer to be tf.train.adam() with a learning rate of 0.0001.
  const optimizer = tf.train.adam(0.0001);

  // Set the following metrics for the callback: 'loss', 'val_loss', 'acc', 'val_acc'.
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc']   
        
  // Create the container for the callback. Set the name to 'Model Training' and 
  // use a height of 1000px for the styles. 

  const container = { name: 'Model Training', styles: { height: '1000px'}};
    
  // Use tfvis.show.fitCallbacks() to setup the callbacks. 
  // Use the container and metrics defined above as the parameters.
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics)
    
        
  // Compile the model using the categoricalCrossentropy loss, and
  // the optimizer you defined above.
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy']});
  tfvis.show.modelSummary({name: 'Model Architecture'}, model);

  // const splited_dataset = dataset.splitTrainTest(0.2);

  let loss = 0;
  await model.fit(dataset.xs, dataset.ys, {
    batchSize: 20,
    epochs: 10,
    shuffle: true,
    validationSplit: 0.1,
    callbacks: fitCallbacks
   });

  alert("Training Done!");

}


function handleButton(elem){
	switch(elem.id){
		case "0":
			rockSamples++;
			document.getElementById("rocksamples").innerText = "Rock samples:" + rockSamples;
			break;
		case "1":
			paperSamples++;
			document.getElementById("papersamples").innerText = "Paper samples:" + paperSamples;
			break;
		case "2":
			scissorsSamples++;
			document.getElementById("scissorssamples").innerText = "Scissors samples:" + scissorsSamples;
			break;  
		case "3":
			spockSamples++;
			document.getElementById("spocksamples").innerText = "Spock samples:" + spockSamples;
			break;
    case "4":
      lizardSamples++;
      document.getElementById("lizardsamples").innerText = "Lizard samples: " + lizardSamples;
      break;
        // Add a case for lizard samples.
        // HINT: Look at the previous cases.
            
        // YOUR CODE HERE
		
            
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(mobilenet.predict(img), label);

}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "I see Rock";
			break;
		case 1:
			predictionText = "I see Paper";
			break;
		case 2:
			predictionText = "I see Scissors";
			break;
		case 3:
			predictionText = "I see Spock";
			break;
    case 4:
      predictionText = "I see Lizard";
      break;
            
        // Add a case for lizard samples.
        // HINT: Look at the previous cases.
            
        // YOUR CODE HERE 
            
	}
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}


function saveModel(){
    model.save('downloads://my_model');
}


async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}


init();