Simple neural network consisting of a few layers and nodes, minimized using a brute force / somewhat genetic approach

The weights are first randomized, then a multiple of "child" NN instances contain the weights of the parent instance, modified to some degree.
Only the best fitting / best scoring NN is used as the next parent.

Sometimes it converges really quickly and nicely, and sometimes it doesn't. 
Extremely inefficient, so I threw in ``rayon`` to make things faster. Your CPU will literally cook itself alive btw.
