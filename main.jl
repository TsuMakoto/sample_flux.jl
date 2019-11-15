include("helper.jl")

# 訓練データの読込
using Flux
using Flux.Data.MNIST
using Flux: onehotbatch
imgs = MNIST.images(:train)
train_X = hcat(float.(vec.(imgs))...)
labels = MNIST.labels(:train)
train_Y = onehotbatch(labels, 0:9)

# 学習モデルの定義
using Flux: Chain
using Flux: Dense
using NNlib: softmax
using NNlib: relu
layeri_1 = Dense(28^2, 100, relu)
layer1_2 = Dense(100, 100, relu)
layer2_o = Dense(100, 10)
model = Chain(layeri_1, layer1_2, layer2_o, softmax)

# 訓練データを32個ずつに分割
using Base.Iterators: partition
batchsize = 32
serial_iterator = partition(1:size(train_Y)[2], batchsize)
train_dataset = map(batch -> (train_X[:, batch], train_Y[:, batch]), serial_iterator)

# run training
using Flux: crossentropy
using Flux: @epochs
using Flux: ADAM
using Flux: train!
loss(x, y) = crossentropy(model(x), y)
opt = ADAM()
epochs = 10
@epochs epochs train!(loss, params(model), train_dataset, opt)

# モデルの保存
# using BSON: @save
pretrained = cpu(model)
weights = params(pretrained)
# @save "/lab/mnist/flux/pretrained.bson" pretrained
# @save "/lab/mnist/flux/weights.bson" weights

# テストデータの読込
imgs = MNIST.images(:test)
test_X = hcat(float.(vec.(imgs))...)
labels = MNIST.labels(:test)

# テストデータの推論
# using BSON: @load
using Statistics: mean
using Flux: onecold
println("Start to evaluate testset")
println("loading pretraiined model")
# @load "/lab/mnist/flux/pretrained.bson" pretrained
model = pretrained
@show mean(onecold(model(test_X)) .== labels)
println("Done")
