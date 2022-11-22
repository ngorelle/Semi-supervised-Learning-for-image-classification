# Semi-supervised-Learning-for-image-classification
Mean teacher methode for image classification


![image](https://user-images.githubusercontent.com/59926481/203343340-5ac9cdcc-89aa-48b4-baf3-74a83b4c7b93.png)

Mean Teacher is a simple method for semi-supervised learning. It consists of the following steps:

Take a supervised architecture and make a copy of it. Let's call the original model the student and the new one the teacher.
At each training step, use the same minibatch as inputs to both the student and the teacher but add random augmentation or noise to the inputs separately.
Add an additional consistency cost between the student and teacher outputs (after softmax).
Let the optimizer update the student weights normally.
Let the teacher weights be an exponential moving average (EMA) of the student weights. That is, after each training step, update the teacher weights a little bit toward the student weights.
