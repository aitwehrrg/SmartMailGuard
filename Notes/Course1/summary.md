## Formula

 n<sub>1</sub>, n<sub>2</sub>, n<sub>3</sub> ... , n<sub>L</sub> = no. of neurons in L layers

n<sub>0</sub> = no. of inputs

parameters : W<sub>l</sub> , b<sub>l</sub>

****

### Forward prop

A<sub>0</sub> = inputs

Z<sub>l</sub> = W<sub>l</sub> @ A<sub>l-1</sub> + b<sub>l</sub>

A<sub>l</sub> = g( Z<sub>l</sub> )

A<sub>L</sub> = outputs

****

### Backward prop

dZ<sub>l</sub> = dA<sub>l</sub> * g'( dZ<sub>l</sub> )

dW<sub>l</sub> = dZ<sub>l</sub> @ A<sub>l-1</sub>.T

dA<sub>l-1</sub> = W<sub>l</sub>.T @ dZ<sub>l</sub>

</br>

W<sub>l</sub> = W<sub>l</sub> - dW<sub>l</sub>

b<sub>l</sub> = b<sub>l</sub> - db<sub>l</sub>

