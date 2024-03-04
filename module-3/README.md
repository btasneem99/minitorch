# minitorch
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/vYQ4W4rf)
# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py

* Task 3.1 + 3.2 

MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/bushratasneem/Documents/code/MLE/mle-
module-3-btasneem99/minitorch/fast_ops.py (154)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/bushratasneem/Documents/code/MLE/mle-module-3-btasneem99/minitorch/fast_ops.py (154) 
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                | 
        out: Storage,                                                        | 
        out_shape: Shape,                                                    | 
        out_strides: Strides,                                                | 
        in_storage: Storage,                                                 | 
        in_shape: Shape,                                                     | 
        in_strides: Strides,                                                 | 
    ) -> None:                                                               | 
        # TODO: Implement for Task 3.1.                                      | 
        if (                                                                 | 
            len(out_strides) != len(in_strides)                              | 
            or (out_strides != in_strides).any()-----------------------------| #0
            or (out_shape != in_shape).any()---------------------------------| #1
        ):                                                                   | 
            for i in prange(len(out)):---------------------------------------| #3
                out_index = np.empty(MAX_DIMS, np.int32)                     | 
                in_index = np.empty(MAX_DIMS, np.int32)                      | 
                to_index(i, out_shape, out_index)                            | 
                broadcast_index(out_index, out_shape, in_shape, in_index)    | 
                out[index_to_position(out_index, out_strides)] = fn(         | 
                    in_storage[index_to_position(in_index, in_strides)]      | 
                )                                                            | 
        else:                                                                | 
            for i in prange(len(out)):---------------------------------------| #2
                out[i] = fn(in_storage[i])                                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #3, #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/bushratasneem/Documents/code/MLE/mle-
module-3-btasneem99/minitorch/fast_ops.py (169) is hoisted out of the parallel 
loop labelled #3 (it will be performed before the loop is executed and reused 
inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/bushratasneem/Documents/code/MLE/mle-
module-3-btasneem99/minitorch/fast_ops.py (170) is hoisted out of the parallel 
loop labelled #3 (it will be performed before the loop is executed and reused 
inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/bushratasneem/Documents/code/MLE/mle-
module-3-btasneem99/minitorch/fast_ops.py (205)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/bushratasneem/Documents/code/MLE/mle-module-3-btasneem99/minitorch/fast_ops.py (205) 
---------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                          | 
        out: Storage,                                                                  | 
        out_shape: Shape,                                                              | 
        out_strides: Strides,                                                          | 
        a_storage: Storage,                                                            | 
        a_shape: Shape,                                                                | 
        a_strides: Strides,                                                            | 
        b_storage: Storage,                                                            | 
        b_shape: Shape,                                                                | 
        b_strides: Strides,                                                            | 
    ) -> None:                                                                         | 
        # TODO: Implement for Task 3.1.                                                | 
        if (                                                                           | 
            len(out_strides) != len(a_strides)                                         | 
            or len(out_strides) != len(b_strides)                                      | 
            or (out_strides != a_strides).any()----------------------------------------| #4
            or (out_strides != b_strides).any()----------------------------------------| #5
            or (out_shape != a_shape).any()--------------------------------------------| #6
            or (out_shape != b_shape).any()--------------------------------------------| #7
        ):                                                                             | 
            for i in prange(len(out)):-------------------------------------------------| #9
                out_index = np.empty(MAX_DIMS, np.int32)                               | 
                a_index = np.empty(MAX_DIMS, np.int32)                                 | 
                b_index = np.empty(MAX_DIMS, np.int32)                                 | 
                to_index(i, out_shape, out_index)                                      | 
                broadcast_index(out_index, out_shape, a_shape, a_index)                | 
                broadcast_index(out_index, out_shape, b_shape, b_index)                | 
                a_data = a_storage[index_to_position(a_index, a_strides)]              | 
                b_data = b_storage[index_to_position(b_index, b_strides)]              | 
                out[index_to_position(out_index, out_strides)] = fn(a_data, b_data)    | 
                                                                                       | 
        else:                                                                          | 
            for i in prange(len(out)):-------------------------------------------------| #8
                out[i] = fn(a_storage[i], b_storage[i])                                | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #9, #8).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/bushratasneem/Documents/code/MLE/mle-
module-3-btasneem99/minitorch/fast_ops.py (226) is hoisted out of the parallel 
loop labelled #9 (it will be performed before the loop is executed and reused 
inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/bushratasneem/Documents/code/MLE/mle-
module-3-btasneem99/minitorch/fast_ops.py (227) is hoisted out of the parallel 
loop labelled #9 (it will be performed before the loop is executed and reused 
inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/bushratasneem/Documents/code/MLE/mle-
module-3-btasneem99/minitorch/fast_ops.py (228) is hoisted out of the parallel 
loop labelled #9 (it will be performed before the loop is executed and reused 
inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/bushratasneem/Documents/code/MLE/mle-
module-3-btasneem99/minitorch/fast_ops.py (262)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/bushratasneem/Documents/code/MLE/mle-module-3-btasneem99/minitorch/fast_ops.py (262) 
-------------------------------------------------------------|loop #ID
    def _reduce(                                             | 
        out: Storage,                                        | 
        out_shape: Shape,                                    | 
        out_strides: Strides,                                | 
        a_storage: Storage,                                  | 
        a_shape: Shape,                                      | 
        a_strides: Strides,                                  | 
        reduce_dim: int,                                     | 
    ) -> None:                                               | 
        # TODO: Implement for Task 3.1.                      | 
        for i in prange(len(out)):---------------------------| #10
            out_index = np.empty(MAX_DIMS, np.int32)         | 
            dim = a_shape[reduce_dim]                        | 
            to_index(i, out_shape, out_index)                | 
            o = index_to_position(out_index, out_strides)    | 
            accum = out[o]                                   | 
            j = index_to_position(out_index, a_strides)      | 
            st = a_strides[reduce_dim]                       | 
            for s in range(dim):                             | 
                accum = fn(accum, a_storage[j])              | 
                j += st                                      | 
            out[o] = accum                                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/bushratasneem/Documents/code/MLE/mle-
module-3-btasneem99/minitorch/fast_ops.py (273) is hoisted out of the parallel 
loop labelled #10 (it will be performed before the loop is executed and reused 
inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/bushratasneem/Documents/code/MLE/mle-
module-3-btasneem99/minitorch/fast_ops.py (288)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/bushratasneem/Documents/code/MLE/mle-module-3-btasneem99/minitorch/fast_ops.py (288) 
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                                                                                                                          | 
    out: Storage,                                                                                                                                                                                     | 
    out_shape: Shape,                                                                                                                                                                                 | 
    out_strides: Strides,                                                                                                                                                                             | 
    a_storage: Storage,                                                                                                                                                                               | 
    a_shape: Shape,                                                                                                                                                                                   | 
    a_strides: Strides,                                                                                                                                                                               | 
    b_storage: Storage,                                                                                                                                                                               | 
    b_shape: Shape,                                                                                                                                                                                   | 
    b_strides: Strides,                                                                                                                                                                               | 
) -> None:                                                                                                                                                                                            | 
    """                                                                                                                                                                                               | 
    NUMBA tensor matrix multiply function.                                                                                                                                                            | 
                                                                                                                                                                                                      | 
    Should work for any tensor shapes that broadcast as long as                                                                                                                                       | 
                                                                                                                                                                                                      | 
    ```                                                                                                                                                                                               | 
    assert a_shape[-1] == b_shape[-2]                                                                                                                                                                 | 
    ```                                                                                                                                                                                               | 
                                                                                                                                                                                                      | 
    Optimizations:                                                                                                                                                                                    | 
                                                                                                                                                                                                      | 
    * Outer loop in parallel                                                                                                                                                                          | 
    * No index buffers or function calls                                                                                                                                                              | 
    * Inner loop should have no global writes, 1 multiply.                                                                                                                                            | 
                                                                                                                                                                                                      | 
                                                                                                                                                                                                      | 
    Args:                                                                                                                                                                                             | 
        out (Storage): storage for `out` tensor                                                                                                                                                       | 
        out_shape (Shape): shape for `out` tensor                                                                                                                                                     | 
        out_strides (Strides): strides for `out` tensor                                                                                                                                               | 
        a_storage (Storage): storage for `a` tensor                                                                                                                                                   | 
        a_shape (Shape): shape for `a` tensor                                                                                                                                                         | 
        a_strides (Strides): strides for `a` tensor                                                                                                                                                   | 
        b_storage (Storage): storage for `b` tensor                                                                                                                                                   | 
        b_shape (Shape): shape for `b` tensor                                                                                                                                                         | 
        b_strides (Strides): strides for `b` tensor                                                                                                                                                   | 
                                                                                                                                                                                                      | 
    Returns:                                                                                                                                                                                          | 
        None : Fills in `out`                                                                                                                                                                         | 
    """                                                                                                                                                                                               | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                                                                                                                            | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                                                                                                                            | 
                                                                                                                                                                                                      | 
    # TODO: Implement for Task 3.2.                                                                                                                                                                   | 
    for n in prange(out_shape[0]):--------------------------------------------------------------------------------------------------------------------------------------------------------------------| #13
        # We then loop through the 1th dimension of a                                                                                                                                                 | 
        for i in prange(out_shape[1]):----------------------------------------------------------------------------------------------------------------------------------------------------------------| #12
            # And loop through the 2th dimension of b                                                                                                                                                 | 
            for j in prange(out_shape[2]):------------------------------------------------------------------------------------------------------------------------------------------------------------| #11
                # We get the absolute positions in a_storage and b_storage by multiplying the indices with the strides                                                                                | 
                a_idx = n * a_batch_stride + i * a_strides[1]                                                                                                                                         | 
                b_idx = n * b_batch_stride + j * b_strides[2]                                                                                                                                         | 
                accum = 0.0                                                                                                                                                                           | 
                # We use the variable accum to store the inner product of matrices a and b while looping through the common dimension k which is the 2th dimension of a and the 1th dimension of b    | 
                for k in range(a_shape[2]):                                                                                                                                                           | 
                    accum += a_storage[a_idx] * b_storage[b_idx]                                                                                                                                      | 
                    # We update the position for both a and b in their common dimension (2th for a and 1th for b) with the help of the strides                                                        | 
                    a_idx += a_strides[2]                                                                                                                                                             | 
                    b_idx += b_strides[1]                                                                                                                                                             | 
                # We calculate the absolute position in out by multiplying the strides with the indices                                                                                               | 
                out[                                                                                                                                                                                  | 
                    n * out_strides[0] + i * out_strides[1] + j * out_strides[2]                                                                                                                      | 
                ] = accum                                                                                                                                                                             | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #13, #12).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--13 is a parallel loop
   +--12 --> rewritten as a serial loop
      +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (parallel)
      +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (serial)
      +--11 (serial)


 
Parallel region 0 (loop #13) had 0 loop(s) fused and 2 loop(s) serialized as 
part of the larger parallel loop (#13).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

* Task 3.4 
The graph shows the runtimes for matrix multiply for FastTensor Backend and GPU Backend for multiples of 64 from 1 to 5,
n = [64, 128, 256, 512, 1024]
![Task 3.4 - Graph](/Task_3.4_Graph.jpeg)
* Task 3.5 
* Split Dataset -

Hidden Layers = 100
Learning Rate = 0.05

Epoch  0  loss  7.505913482547551 correct 25 time per epoch 6.777040243148804
Epoch  10  loss  6.051106249327841 correct 34 time per epoch 2.162961483001709
Epoch  20  loss  7.254476433754651 correct 35 time per epoch 1.4905102252960205
Epoch  30  loss  4.300923528710416 correct 39 time per epoch 1.4901995658874512
Epoch  40  loss  4.899134717908462 correct 41 time per epoch 1.5762412548065186
Epoch  50  loss  3.4895292422564474 correct 44 time per epoch 2.5652408599853516
Epoch  60  loss  4.890521179404486 correct 42 time per epoch 2.4761879444122314
Epoch  70  loss  5.812549789159487 correct 43 time per epoch 1.5583231449127197
Epoch  80  loss  3.556024636031129 correct 45 time per epoch 1.5541036128997803
Epoch  90  loss  2.6373101934183296 correct 45 time per epoch 1.4934196472167969
Epoch  100  loss  2.2947926652616304 correct 45 time per epoch 2.499835729598999
Epoch  110  loss  1.9062428841684518 correct 45 time per epoch 2.2837212085723877
Epoch  120  loss  2.770168610772639 correct 47 time per epoch 2.5837905406951904
Epoch  130  loss  0.8375917892723126 correct 46 time per epoch 2.3084475994110107
Epoch  140  loss  4.448098466020683 correct 43 time per epoch 1.5193006992340088
Epoch  150  loss  1.5223569189714399 correct 48 time per epoch 1.4966504573822021
Epoch  160  loss  2.787634152140921 correct 47 time per epoch 1.6096351146697998
Epoch  170  loss  2.190301880596434 correct 48 time per epoch 2.5513393878936768
Epoch  180  loss  0.46722806878448436 correct 47 time per epoch 2.5713164806365967
Epoch  190  loss  0.896272873750752 correct 47 time per epoch 2.3057546615600586
Epoch  200  loss  0.20047025532981416 correct 47 time per epoch 1.589780330657959
Epoch  210  loss  0.9483428937705637 correct 47 time per epoch 1.508279800415039
Epoch  220  loss  1.183982390545984 correct 47 time per epoch 1.5831410884857178
Epoch  230  loss  1.4485910781458324 correct 47 time per epoch 2.543886661529541
Epoch  240  loss  1.5440762988886665 correct 48 time per epoch 2.76237416267395
Epoch  250  loss  2.0747417118282008 correct 48 time per epoch 2.266256093978882
Epoch  260  loss  1.154952512897103 correct 48 time per epoch 1.5743110179901123
Epoch  270  loss  0.943347237480757 correct 50 time per epoch 1.508882761001587
Epoch  280  loss  5.441928335253375 correct 45 time per epoch 1.61090087890625
Epoch  290  loss  1.1077993179285417 correct 47 time per epoch 2.571500301361084
Epoch  300  loss  1.02404243719229 correct 48 time per epoch 2.563645601272583
Epoch  310  loss  0.2896467410959623 correct 48 time per epoch 2.4476542472839355
Epoch  320  loss  1.2398778742372645 correct 49 time per epoch 1.572911262512207
Epoch  330  loss  0.4793083111338797 correct 48 time per epoch 1.5006752014160156
Epoch  340  loss  1.9818188012749385 correct 44 time per epoch 1.5324764251708984
Epoch  350  loss  1.941770510680042 correct 47 time per epoch 2.551004409790039
Epoch  360  loss  0.7841935539774849 correct 48 time per epoch 2.6688473224639893
Epoch  370  loss  3.5798216661324775 correct 49 time per epoch 2.2979109287261963
Epoch  380  loss  2.2278196602006606 correct 48 time per epoch 1.53700590133667
Epoch  390  loss  2.1662588350241716 correct 49 time per epoch 1.5167286396026611
Epoch  400  loss  0.7991303601350537 correct 49 time per epoch 1.6141085624694824
Epoch  410  loss  0.7357566436629719 correct 48 time per epoch 2.544887065887451
Epoch  420  loss  1.4503705203730561 correct 48 time per epoch 2.57879900932312
Epoch  430  loss  2.167696378458644 correct 48 time per epoch 2.6662046909332275
Epoch  440  loss  1.5586811779313172 correct 48 time per epoch 1.895979642868042
Epoch  450  loss  0.5568337675117115 correct 48 time per epoch 1.5007860660552979
Epoch  460  loss  1.7347227267684926 correct 48 time per epoch 1.5034122467041016
Epoch  470  loss  1.676235513849804 correct 47 time per epoch 1.8157451152801514
Epoch  480  loss  2.204767243459577 correct 48 time per epoch 2.621389389038086
Epoch  490  loss  1.9463563252000435 correct 48 time per epoch 2.451608180999756

Hidden Layers = 200
Learning Rate = 0.05

Epoch  0  loss  7.67483250189381 correct 20 time per epoch 6.045195579528809
Epoch  10  loss  2.959734409580154 correct 41 time per epoch 2.0042271614074707
Epoch  20  loss  3.8268243088970926 correct 45 time per epoch 1.983854055404663
Epoch  30  loss  1.420014142437712 correct 44 time per epoch 2.0930347442626953
Epoch  40  loss  1.0611496986972522 correct 48 time per epoch 2.6644651889801025
Epoch  50  loss  2.0672525349616 correct 47 time per epoch 3.142333984375
Epoch  60  loss  0.4879975386864934 correct 50 time per epoch 3.451892852783203
Epoch  70  loss  0.7466267115772305 correct 50 time per epoch 3.5907363891601562
Epoch  80  loss  1.3327389723007106 correct 49 time per epoch 2.1036977767944336
Epoch  90  loss  0.8293402363938553 correct 50 time per epoch 2.0150725841522217
Epoch  100  loss  0.6540365492144524 correct 50 time per epoch 2.179373025894165
Epoch  110  loss  0.7970451206776804 correct 50 time per epoch 2.983781337738037
Epoch  120  loss  0.6706369044920524 correct 50 time per epoch 3.38108229637146
Epoch  130  loss  0.2109051906808444 correct 50 time per epoch 3.4162776470184326
Epoch  140  loss  0.5847277699112762 correct 50 time per epoch 3.460439920425415
Epoch  150  loss  0.7141252286916806 correct 50 time per epoch 2.083928346633911
Epoch  160  loss  0.4134731918019851 correct 50 time per epoch 2.0591719150543213
Epoch  170  loss  0.37537485254331077 correct 50 time per epoch 2.0306789875030518
Epoch  180  loss  0.28357193711014694 correct 50 time per epoch 2.1417553424835205
Epoch  190  loss  0.14878750424705772 correct 50 time per epoch 2.491027593612671
Epoch  200  loss  0.24695470168937544 correct 50 time per epoch 3.415877342224121
Epoch  210  loss  0.2542580134215729 correct 50 time per epoch 3.3213553428649902
Epoch  220  loss  0.4490653979783866 correct 50 time per epoch 3.386186361312866
Epoch  230  loss  0.4823777348146474 correct 50 time per epoch 2.309030294418335
Epoch  240  loss  0.039281444641853756 correct 50 time per epoch 2.026958703994751
Epoch  250  loss  0.42192334531576475 correct 50 time per epoch 2.024064540863037
Epoch  260  loss  0.12968965167703894 correct 50 time per epoch 2.0929267406463623
Epoch  270  loss  0.3044178356676292 correct 50 time per epoch 2.019240140914917
Epoch  280  loss  0.284729246394867 correct 50 time per epoch 3.5082311630249023
Epoch  290  loss  0.16842420254994703 correct 50 time per epoch 3.289310932159424
Epoch  300  loss  0.01927108271897309 correct 50 time per epoch 3.443572759628296
Epoch  310  loss  0.42887754164772107 correct 50 time per epoch 3.403177499771118
Epoch  320  loss  0.36407821958688 correct 50 time per epoch 2.029221773147583
Epoch  330  loss  0.10808369860780753 correct 50 time per epoch 2.0179147720336914
Epoch  340  loss  0.33975000127764304 correct 50 time per epoch 2.1202118396759033
Epoch  350  loss  0.32885897100620903 correct 50 time per epoch 2.033996343612671
Epoch  360  loss  0.06676358525919579 correct 50 time per epoch 2.079897165298462
Epoch  370  loss  0.037754427236820085 correct 50 time per epoch 3.145068645477295
Epoch  380  loss  0.11536208656162736 correct 50 time per epoch 3.4537253379821777
Epoch  390  loss  0.2420447519848276 correct 50 time per epoch 3.3157753944396973
Epoch  400  loss  0.0465844278064601 correct 50 time per epoch 3.3564677238464355
Epoch  410  loss  0.2187085283711161 correct 50 time per epoch 2.2079105377197266
Epoch  420  loss  0.0031607683281204154 correct 50 time per epoch 2.083223819732666
Epoch  430  loss  0.09717004082502997 correct 50 time per epoch 2.0459835529327393
Epoch  440  loss  0.07490784101893645 correct 50 time per epoch 3.565589427947998
Epoch  450  loss  0.033027298365145674 correct 50 time per epoch 3.1615712642669678
Epoch  460  loss  0.09195880564924293 correct 50 time per epoch 3.3302500247955322
Epoch  470  loss  0.11434220146538863 correct 50 time per epoch 3.371593713760376
Epoch  480  loss  0.004650032683454517 correct 50 time per epoch 3.0677366256713867
Epoch  490  loss  0.18101893440595165 correct 50 time per epoch 2.050042152404785


* Simple Dataset -

Hidden Layers = 100
Learning Rate = 0.05

Epoch  0  loss  5.202425734671244 correct 37 time per epoch 5.963679075241089
Epoch  10  loss  1.2137495353286074 correct 45 time per epoch 2.5939853191375732
Epoch  20  loss  3.662095968250769 correct 47 time per epoch 3.0751140117645264
Epoch  30  loss  1.190092231055324 correct 49 time per epoch 2.545605421066284
Epoch  40  loss  1.1568245754105249 correct 48 time per epoch 2.060941219329834
Epoch  50  loss  0.755719044140146 correct 50 time per epoch 1.5194602012634277
Epoch  60  loss  0.324737805926249 correct 50 time per epoch 1.55049467086792
Epoch  70  loss  0.10397900847062866 correct 50 time per epoch 1.512904405593872
Epoch  80  loss  0.4814605326300677 correct 50 time per epoch 2.6789379119873047
Epoch  90  loss  0.13097006455722776 correct 50 time per epoch 2.5713016986846924
Epoch  100  loss  0.21749607804178842 correct 50 time per epoch 2.208028793334961
Epoch  110  loss  0.14835716170056962 correct 50 time per epoch 1.5327568054199219
Epoch  120  loss  0.6591890599746719 correct 50 time per epoch 1.5334808826446533
Epoch  130  loss  0.14157225878102514 correct 50 time per epoch 1.529754877090454
Epoch  140  loss  0.052540486801392144 correct 50 time per epoch 2.6597185134887695
Epoch  150  loss  0.5579034530250171 correct 50 time per epoch 2.5824294090270996
Epoch  160  loss  0.1616499618256698 correct 50 time per epoch 2.5536534786224365
Epoch  170  loss  0.06996959489917065 correct 50 time per epoch 2.313239574432373
Epoch  180  loss  0.48753643933399404 correct 50 time per epoch 1.5754668712615967
Epoch  190  loss  0.11932978668619433 correct 50 time per epoch 1.5109052658081055
Epoch  200  loss  0.8298604432414247 correct 50 time per epoch 1.548344612121582
Epoch  210  loss  0.12287530834026386 correct 50 time per epoch 2.571669101715088
Epoch  220  loss  0.044577984544680885 correct 50 time per epoch 2.671267509460449
Epoch  230  loss  0.17096414560328 correct 50 time per epoch 2.3320391178131104
Epoch  240  loss  0.7810070059467771 correct 50 time per epoch 1.5240437984466553
Epoch  250  loss  0.05010104635097317 correct 50 time per epoch 1.5503811836242676
Epoch  260  loss  0.3065149531991896 correct 50 time per epoch 1.5947294235229492
Epoch  270  loss  0.12128084181763726 correct 50 time per epoch 2.0778300762176514
Epoch  280  loss  0.005812603603864916 correct 50 time per epoch 2.556952714920044
Epoch  290  loss  0.13278037678944216 correct 50 time per epoch 2.526380777359009
Epoch  300  loss  0.17740361424268639 correct 50 time per epoch 2.3442490100860596
Epoch  310  loss  0.12857692722323066 correct 50 time per epoch 1.5033836364746094
Epoch  320  loss  0.0016928701937145012 correct 50 time per epoch 2.5573489665985107
Epoch  330  loss  0.03635606186488875 correct 50 time per epoch 1.5121259689331055
Epoch  340  loss  0.07351443909172169 correct 50 time per epoch 1.587975025177002
Epoch  350  loss  0.17343614468523608 correct 50 time per epoch 2.116799831390381
Epoch  360  loss  0.19755384465104733 correct 50 time per epoch 2.600477457046509
Epoch  370  loss  0.23084856221680133 correct 50 time per epoch 2.374260425567627
Epoch  380  loss  0.0034499352108109532 correct 50 time per epoch 1.5654146671295166
Epoch  390  loss  0.022086944579945998 correct 50 time per epoch 1.5319032669067383
Epoch  400  loss  0.1908033559461752 correct 50 time per epoch 1.5211834907531738
Epoch  410  loss  0.08746256886117575 correct 50 time per epoch 2.1413514614105225
Epoch  420  loss  0.2917660780999989 correct 50 time per epoch 2.6775906085968018
Epoch  430  loss  0.35000183618607916 correct 50 time per epoch 2.4478037357330322
Epoch  440  loss  0.028809225049208416 correct 50 time per epoch 1.5184295177459717
Epoch  450  loss  0.010834029774191596 correct 50 time per epoch 1.5068557262420654
Epoch  460  loss  0.00019337238750256137 correct 50 time per epoch 1.5104401111602783
Epoch  470  loss  0.03659018819031481 correct 50 time per epoch 2.558830976486206
Epoch  480  loss  0.06592652615189527 correct 50 time per epoch 2.667224407196045
Epoch  490  loss  0.260223004235456 correct 50 time per epoch 2.3873445987701416

Hidden Layers = 200
Learning Rate = 0.05

Epoch  0  loss  3.4502206881638777 correct 41 time per epoch 7.686286926269531
Epoch  10  loss  2.632191137916986 correct 49 time per epoch 3.6838996410369873
Epoch  20  loss  0.8800304827235578 correct 48 time per epoch 2.2061946392059326
Epoch  30  loss  2.3708078632064327 correct 47 time per epoch 2.110311269760132
Epoch  40  loss  0.3085428352173215 correct 50 time per epoch 2.194211721420288
Epoch  50  loss  0.3124820443226134 correct 50 time per epoch 2.197265625
Epoch  60  loss  0.3979333814432573 correct 50 time per epoch 2.4811103343963623
Epoch  70  loss  0.5954731815496923 correct 50 time per epoch 3.340674638748169
Epoch  80  loss  0.774689847303232 correct 50 time per epoch 3.6190574169158936
Epoch  90  loss  0.36629011211863916 correct 50 time per epoch 3.235671043395996
Epoch  100  loss  0.33803667781903485 correct 50 time per epoch 2.178685188293457
Epoch  110  loss  0.5695774011985537 correct 50 time per epoch 2.1016604900360107
Epoch  120  loss  0.0906222137457951 correct 50 time per epoch 2.062227487564087
Epoch  130  loss  0.008433108318123661 correct 50 time per epoch 2.742741346359253
Epoch  140  loss  0.5654360186422943 correct 50 time per epoch 3.56882905960083
Epoch  150  loss  0.2205592168365583 correct 50 time per epoch 3.5212650299072266
Epoch  160  loss  0.19943375467873178 correct 50 time per epoch 2.8572537899017334
Epoch  170  loss  0.6688750958666534 correct 50 time per epoch 2.415726661682129
Epoch  180  loss  0.1298017099495887 correct 50 time per epoch 2.240492343902588
Epoch  190  loss  0.03751362996122673 correct 50 time per epoch 2.119309186935425
Epoch  200  loss  0.2090749419590706 correct 50 time per epoch 2.626378059387207
Epoch  210  loss  0.4678150222537398 correct 50 time per epoch 3.4394514560699463
Epoch  220  loss  0.3967457982335666 correct 50 time per epoch 3.617396593093872
Epoch  230  loss  0.11655527007753123 correct 50 time per epoch 2.7421672344207764
Epoch  240  loss  0.5297668457829706 correct 50 time per epoch 2.1593832969665527
Epoch  250  loss  0.06072009171651518 correct 50 time per epoch 2.10966420173645
Epoch  260  loss  0.5466830328700076 correct 50 time per epoch 2.312628984451294
Epoch  270  loss  0.05307371380584461 correct 50 time per epoch 3.4388632774353027
Epoch  280  loss  0.0010460164910991887 correct 50 time per epoch 3.8079848289489746
Epoch  290  loss  0.08852010231277144 correct 50 time per epoch 3.127269983291626
Epoch  300  loss  0.03487887486027624 correct 50 time per epoch 2.187331438064575
Epoch  310  loss  0.6679897044920717 correct 50 time per epoch 2.1864867210388184
Epoch  320  loss  0.07087760333964382 correct 50 time per epoch 3.292070150375366
Epoch  330  loss  0.3048575846472281 correct 50 time per epoch 3.5600028038024902
Epoch  340  loss  0.35896430182618067 correct 50 time per epoch 3.6785106658935547
Epoch  350  loss  0.0981835281702209 correct 50 time per epoch 2.161018133163452
Epoch  360  loss  0.7028516368620383 correct 50 time per epoch 2.209697961807251
Epoch  370  loss  0.008421751946233371 correct 50 time per epoch 2.1525955200195312
Epoch  380  loss  0.10802487472489097 correct 50 time per epoch 3.5337600708007812
Epoch  390  loss  0.09629262196572827 correct 50 time per epoch 4.1597900390625
Epoch  400  loss  0.5355454833023868 correct 50 time per epoch 2.164289951324463
Epoch  410  loss  0.0036540382782977426 correct 50 time per epoch 2.1762313842773438
Epoch  420  loss  0.3427888459307795 correct 50 time per epoch 3.363194704055786
Epoch  430  loss  0.03879691399851351 correct 50 time per epoch 3.61466908454895
Epoch  440  loss  0.31865522740052 correct 50 time per epoch 3.6267571449279785
Epoch  450  loss  0.18279490258514433 correct 50 time per epoch 2.1444904804229736
Epoch  460  loss  0.41292574333771426 correct 50 time per epoch 2.13189435005188
Epoch  470  loss  0.2980720497232088 correct 50 time per epoch 2.163560152053833
Epoch  480  loss  0.026051341333754877 correct 50 time per epoch 3.3322670459747314
Epoch  490  loss  0.43943533252856776 correct 50 time per epoch 3.529611349105835

* Xor Dataset -

Hidden Layers = 100
Learning Rate = 0.05

Epoch  0  loss  6.673912157512884 correct 28 time per epoch 4.5590150356292725
Epoch  10  loss  6.279792030726778 correct 32 time per epoch 1.5414903163909912
Epoch  20  loss  4.481160854672797 correct 41 time per epoch 1.610377311706543
Epoch  30  loss  5.710580340301492 correct 40 time per epoch 1.5645716190338135
Epoch  40  loss  2.945331481251196 correct 42 time per epoch 2.7039430141448975
Epoch  50  loss  6.026729576872266 correct 34 time per epoch 2.5774154663085938
Epoch  60  loss  3.9461819945040286 correct 40 time per epoch 2.337465286254883
Epoch  70  loss  4.1104664601192 correct 37 time per epoch 1.4966843128204346
Epoch  80  loss  3.4901507005860792 correct 46 time per epoch 1.5772409439086914
Epoch  90  loss  3.148169366124143 correct 44 time per epoch 1.5568995475769043
Epoch  100  loss  2.612526331164385 correct 49 time per epoch 2.765451192855835
Epoch  110  loss  2.424995921925366 correct 40 time per epoch 2.7502684593200684
Epoch  120  loss  2.069863681366975 correct 45 time per epoch 2.585413932800293
Epoch  130  loss  2.6231574905490813 correct 50 time per epoch 2.3589630126953125
Epoch  140  loss  3.047680022649931 correct 47 time per epoch 1.6358723640441895
Epoch  150  loss  3.1383328361667022 correct 48 time per epoch 1.605588436126709
Epoch  160  loss  5.794514745581384 correct 43 time per epoch 1.615661859512329
Epoch  170  loss  1.412983582703429 correct 49 time per epoch 2.4988186359405518
Epoch  180  loss  2.5154958339157236 correct 50 time per epoch 1.650935411453247
Epoch  190  loss  2.4090893898066246 correct 46 time per epoch 2.6950435638427734
Epoch  200  loss  1.0516449372299534 correct 50 time per epoch 2.628122568130493
Epoch  210  loss  2.983046840116401 correct 48 time per epoch 2.4670541286468506
Epoch  220  loss  1.9382784403074638 correct 49 time per epoch 1.8238499164581299
Epoch  230  loss  2.1962365729207196 correct 50 time per epoch 1.5650825500488281
Epoch  240  loss  2.715393502882293 correct 50 time per epoch 1.5261123180389404
Epoch  250  loss  1.113719226397479 correct 47 time per epoch 1.5115585327148438
Epoch  260  loss  0.9396421975365344 correct 48 time per epoch 2.7053260803222656
Epoch  270  loss  2.0636987419837363 correct 50 time per epoch 2.6147284507751465
Epoch  280  loss  0.842920700576075 correct 50 time per epoch 2.2750344276428223
Epoch  290  loss  1.5720424980252108 correct 50 time per epoch 1.5106616020202637
Epoch  300  loss  1.1544392200111662 correct 50 time per epoch 1.5843682289123535
Epoch  310  loss  1.125108912678594 correct 50 time per epoch 2.7440309524536133
Epoch  320  loss  0.5507795976481857 correct 49 time per epoch 1.5506832599639893
Epoch  330  loss  1.378328619863858 correct 47 time per epoch 1.5634541511535645
Epoch  340  loss  0.2211044584895029 correct 50 time per epoch 2.802432060241699
Epoch  350  loss  0.6378886028818704 correct 49 time per epoch 2.6582632064819336
Epoch  360  loss  1.828931793237972 correct 46 time per epoch 2.3411378860473633
Epoch  370  loss  2.3357527998314476 correct 49 time per epoch 1.581456184387207
Epoch  380  loss  1.2546984315530865 correct 50 time per epoch 1.6430130004882812
Epoch  390  loss  1.1243693180149639 correct 50 time per epoch 1.5654613971710205
Epoch  400  loss  0.8537350570509553 correct 49 time per epoch 1.6877236366271973
Epoch  410  loss  2.118565224097824 correct 50 time per epoch 2.6131317615509033
Epoch  420  loss  0.9907751839918262 correct 50 time per epoch 2.73506236076355
Epoch  430  loss  1.023508366415737 correct 50 time per epoch 2.729548454284668
Epoch  440  loss  1.422812728052841 correct 50 time per epoch 2.3346118927001953
Epoch  450  loss  1.0159123869823714 correct 49 time per epoch 2.6153526306152344
Epoch  460  loss  0.874644160087173 correct 49 time per epoch 1.503777027130127
Epoch  470  loss  0.7546788962563751 correct 50 time per epoch 1.5289227962493896
Epoch  480  loss  0.7664525338835348 correct 50 time per epoch 1.5496459007263184
Epoch  490  loss  0.42775038057037734 correct 50 time per epoch 2.3134076595306396

Hidden Layers = 200
Learning Rate = 0.05

Epoch  0  loss  6.319916043797808 correct 26 time per epoch 6.122441530227661
Epoch  10  loss  8.551791966403968 correct 37 time per epoch 2.0939865112304688
Epoch  20  loss  3.493790114740241 correct 40 time per epoch 2.128053665161133
Epoch  30  loss  5.662887738113345 correct 41 time per epoch 2.0863935947418213
Epoch  40  loss  4.099493588586535 correct 42 time per epoch 3.123202323913574
Epoch  50  loss  1.0893259090565859 correct 46 time per epoch 3.6461334228515625
Epoch  60  loss  1.902689780182945 correct 45 time per epoch 3.621997117996216
Epoch  70  loss  3.1345703227058386 correct 45 time per epoch 2.3948793411254883
Epoch  80  loss  3.1787942174107666 correct 42 time per epoch 2.1630218029022217
Epoch  90  loss  2.0762445677570986 correct 47 time per epoch 2.1024935245513916
Epoch  100  loss  1.4047195755223993 correct 48 time per epoch 2.1688175201416016
Epoch  110  loss  2.0449662624654437 correct 49 time per epoch 3.2686140537261963
Epoch  120  loss  1.3402625912849353 correct 48 time per epoch 3.4808247089385986
Epoch  130  loss  1.846966081227784 correct 47 time per epoch 3.559096574783325
Epoch  140  loss  1.1463735670259712 correct 46 time per epoch 2.156113624572754
Epoch  150  loss  0.2638588251751097 correct 47 time per epoch 2.0717906951904297
Epoch  160  loss  1.700509885710696 correct 48 time per epoch 2.104764699935913
Epoch  170  loss  1.4187861955722845 correct 48 time per epoch 2.21811842918396
Epoch  180  loss  0.9150149269974811 correct 49 time per epoch 3.418630361557007
Epoch  190  loss  0.6262924287579628 correct 49 time per epoch 3.3852195739746094
Epoch  200  loss  0.5815240262878509 correct 49 time per epoch 3.45351243019104
Epoch  210  loss  1.799672453227231 correct 49 time per epoch 3.2063984870910645
Epoch  220  loss  0.9741687937239095 correct 48 time per epoch 2.1606717109680176
Epoch  230  loss  1.4930499430892017 correct 48 time per epoch 2.1234264373779297
Epoch  240  loss  1.4709835143495467 correct 49 time per epoch 2.131690740585327
Epoch  250  loss  1.8046980544697822 correct 49 time per epoch 3.2640180587768555
Epoch  260  loss  1.3826142092469595 correct 49 time per epoch 3.5980706214904785
Epoch  270  loss  0.33755084711398886 correct 49 time per epoch 3.516016721725464
Epoch  280  loss  0.8014726983870604 correct 49 time per epoch 2.352670907974243
Epoch  290  loss  0.19371873503847653 correct 49 time per epoch 3.5520095825195312
Epoch  300  loss  3.038410026868683 correct 46 time per epoch 2.6939854621887207
Epoch  310  loss  1.6924070238916555 correct 49 time per epoch 3.410940408706665
Epoch  320  loss  0.20278989966610853 correct 48 time per epoch 3.5002763271331787
Epoch  330  loss  0.901266588207164 correct 50 time per epoch 2.9748830795288086
Epoch  340  loss  1.1679440091690219 correct 49 time per epoch 2.178382635116577
Epoch  350  loss  0.6431391414687685 correct 50 time per epoch 3.5883665084838867
Epoch  360  loss  0.3902388687111857 correct 49 time per epoch 3.2016184329986572
Epoch  370  loss  1.712996755700463 correct 48 time per epoch 3.5756421089172363
Epoch  380  loss  0.444104844363021 correct 48 time per epoch 3.676804304122925
Epoch  390  loss  0.8371537923845898 correct 49 time per epoch 2.1662955284118652
Epoch  400  loss  0.13595398474439244 correct 49 time per epoch 2.1441304683685303
Epoch  410  loss  0.11125794563985475 correct 48 time per epoch 2.146033525466919
Epoch  420  loss  0.7028330095546154 correct 48 time per epoch 2.4167752265930176
Epoch  430  loss  0.7711555650857393 correct 49 time per epoch 3.465596914291382
Epoch  440  loss  0.5475169106037439 correct 48 time per epoch 3.5725412368774414
Epoch  450  loss  0.9564964123795263 correct 50 time per epoch 2.581300973892212
Epoch  460  loss  1.3056862602127326 correct 49 time per epoch 2.164945125579834
Epoch  470  loss  0.13343969018940102 correct 49 time per epoch 2.1264970302581787
Epoch  480  loss  0.17813029458486265 correct 49 time per epoch 2.942422389984131
Epoch  490  loss  1.3213740068660924 correct 48 time per epoch 3.4964759349823
