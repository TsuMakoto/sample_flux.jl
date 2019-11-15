function mnist_show(data, x, y)
  for i ∈ 0:x-1
    for j ∈ 0:y-1
      print(data[i + 28 * j + 1] > 0 ? "*" : " ")
    end
    println()
  end
end
