-- MNIST example taken from
-- https://github.com/torch/demos/tree/master/train-a-digit-classifier

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'

full  = false
learningRate = 0.05
batchSize = 10
momentum = 0
coefL1 = 0
coefL2 = 0
threads = 4


-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)


-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats, for SGD
torch.setdefaulttensortype('torch.FloatTensor')

classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {32,32}

-- define model to train
model = nn.Sequential()
-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- stage 3 : standard 2-layer MLP:
model:add(nn.Reshape(64*2*2))
model:add(nn.Linear(64*2*2, 200))
model:add(nn.Tanh())
model:add(nn.Linear(200, #classes))

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<mnist> using model:')
print(model)

model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

--get first batch
local batch = torch.Tensor(batchSize,1,geometry[1],geometry[2])
local k = 1
for i = 1,batchSize do
   batch[k] = trainData[i][1]:clone()
   k = k + 1
end
model  = require('lsuv')(model, batch, nil, nil)

if full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
else
   nbTrainingPatches = 2000
   nbTestingPatches = 1000
   print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end

----------------------------------------------------------------------
-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,dataset:size(),batchSize do
      -- create mini batch
      local inputs = torch.Tensor(batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(batchSize)
      local k = 1
      for i = t,math.min(t+batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         -- penalties (L1 and L2):
         if coefL1 ~= 0 or coefL2 ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign

            -- Loss:
            f = f + coefL1 * norm(parameters,1)
            f = f + coefL2 * norm(parameters,2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(coefL1) + parameters:clone():mul(coefL2) )
         end

         -- update confusion
         for i = 1,batchSize do
            confusion:add(outputs[i], targets[i])
         end

         -- return f and df/dX
         return f,gradParameters
      end

     -- Perform SGD step:
     sgdState = sgdState or {
        learningRate = learningRate,
        momentum = momentum,
        learningRateDecay = 5e-7
     }
     optim.sgd(feval, parameters, sgdState)

     -- disp progress
     xlua.progress(t, dataset:size())

   end

   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   confusion:zero()

   -- next epoch
   epoch = epoch + 1
end

function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size(),batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.Tensor(batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(batchSize)
      local k = 1
      for i = t,math.min(t+batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- test samples
      local preds = model:forward(inputs)

      -- confusion:
      for i = 1,batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   confusion:zero()
end

----------------------------------------------------------------------
while true do
   train(trainData)
   test(testData)
end
