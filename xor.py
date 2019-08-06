import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sig_der(x):
	return x*(1-x)

x = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 0, 1, 0], [0, 1, 1, 0, 0], [0, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 0, 0, 1, 0], [1, 0, 1, 0, 0], [1, 0, 1, 1, 0], [1, 1, 0, 0, 0], [1, 1, 0, 1, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 1, 1], [0, 1, 0, 0, 1], [0, 1, 0, 1, 1], [0, 1, 1, 0, 1], [0, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 1, 1], [1, 1, 0, 0, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 1]])
y = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 1], [0, 0], [1, 1], [1, 0], [1, 0], [1, 1], [0, 0], [0, 1], [1, 1], [1, 0], [0, 1], [0, 0], [1, 1], [1, 0], [0, 1], [0, 0], [1, 0], [1, 1], [0, 0], [0, 1], [0, 1], [0, 0], [1, 1], [1, 0], [0, 0], [0, 1], [1, 0], [1, 1]])

out_weights = np.random.rand(32, 2)
hidden_weights = np.random.rand(5, 32)
hidden_bias = np.random.rand(1, 32)
out_bias = np.random.rand(1, 2)

inputlayerneurons, hiddenlayerneurons, outputneurons = 5, 32, 2

epochs = 100000
lr = 0.1

for i in range(epochs):
	#########forwardPropagation############
	z1 = np.dot(x, hidden_weights)
	z1 = z1 + hidden_bias
	a1 = sigmoid(z1)
	# print(z1)

	output_z = np.dot(a1, out_weights)
	output_z = output_z + out_bias 
	output = sigmoid(output_z)

	#########backwardPropagation###########

	d_out_weights_error1 = y - output
	d_out_weights_error2 = sig_der(output)
	d_out_weights = np.dot(a1.T, (d_out_weights_error1*d_out_weights_error2))
	# print(d_out_weights)

	d_hidden_weights_err1 = d_out_weights_error1*d_out_weights_error2
	d_hidden_weights_err1 = np.dot(d_hidden_weights_err1, out_weights.T)
	d_hidden_weights_err2 = sig_der(a1)
	d_hidden_weights_err = d_hidden_weights_err1*d_hidden_weights_err2
	d_hidden_weights_err = np.dot(x.T, d_hidden_weights_err)

	d_bias_output = np.sum(d_out_weights_error1*d_out_weights_error2, keepdims = True, axis = 0)
	d_bias_hidden = np.sum(d_hidden_weights_err1*d_hidden_weights_err2, keepdims = True, axis = 0)

	out_weights+=d_out_weights*lr
	hidden_weights+=d_hidden_weights_err*lr

	out_bias+=d_bias_output*lr
	hidden_bias+=d_bias_hidden*lr

print(out_weights)
print(hidden_weights)
print(hidden_bias)
print(out_bias)
np.save('hidden_weights.npy', hidden_weights)
np.save('hidden_bias.npy', hidden_bias)
np.save('out_weights.npy', out_weights)
np.save('out_bias.npy', out_bias)
for i in x:
	z1 = np.dot(i, hidden_weights)
	z1 = z1 + hidden_bias
	a1 = sigmoid(z1)
	# print(z1)

	output_z = np.dot(a1, out_weights)
	output_z = output_z + out_bias 
	output = sigmoid(output_z)
	if(output[0][0]>=0.5):
		output[0][0] = 1
	else:
		output[0][0] = 0
	if(output[0][1]>=0.5):
		output[0][1] = 1
	else:
		output[0][1] = 0
	print(output)