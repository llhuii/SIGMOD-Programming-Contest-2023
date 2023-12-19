# knn graph 优化

题目：为一组高维向量构建k近邻图. k近邻图是一个有向图，其中的每一个点和其k个近似最近邻相连。每一个点代表着一个向量，它们之间的相似度用 [欧氏距离](https://baike.baidu.com/item/%E6%AC%A7%E5%87%A0%E9%87%8C%E5%BE%97%E5%BA%A6%E9%87%8F/1274107#:~:text=%E6%AC%A7%E5%87%A0%E9%87%8C%E5%BE%97%E5%BA%A6%E9%87%8F%EF%BC%88euclidean%20metric%EF%BC%89%EF%BC%88%E4%B9%9F%E7%A7%B0,%E4%B9%8B%E9%97%B4%E7%9A%84%E5%AE%9E%E9%99%85%E8%B7%9D%E7%A6%BB%E3%80%82) 来衡量。

规模：总共1千万个点， 每个点200维度。k=100

网址：[http://11.165.116.46/index.shtml](http://11.165.116.46/index.shtml)

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ybEnBBQw2JwPnP13/img/1c4a537e-484b-40d0-bb24-8af68ebfad5d.png)

> 图中num\_dimension说的是100，但是实际数据集是200

---

最终结果：

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/ybEnBBQw2JwPnP13/img/60095102-42c2-44fd-a84f-052047af59c6.png)

## 我们的优化点：

**整体思路：**本题的关键在于在有限的CPU、内存和时间内取得更好的结果，因此通过空间换时间以及内存压缩，缩短每轮迭代的耗时，逐个阶段调优，增大超参和迭代次数以提高召回率。

下面将按照阶段依次说明我们的优化点。

knn图的构造整体流程如下：

1.  init阶段：加载数据集，初始化，预计算等
    
2.  多次迭代更新邻居表，每次迭代分为两步：
    
    1.  join阶段： local join 每个点的邻居表，更新邻居的邻居表
        
    2.  update阶段：更新下一次join所需的邻居表
        
3.  生成knn图
    
```c
# 1. init phase 
    
for iter {
    
  # 2. join phase
  {
    for each node {
      local join 每个点的邻居表，更新邻居的邻居表 
    }
  }
    
  # 3. update phase
  {
    for each node {
      更新下一次join所需的邻居表 
    }
  }
  
}
# 4. 生成knn图
```


### **join阶段的优化**

用于构建新的邻居候选列表, 每个节点包括：

nn：正向邻居

*   nn\_new: 本次迭代新加的邻居
    
*   nn\_old: 原有 knng 中的邻居
    

rnn：反向邻居（哪些节点的邻居包含该节点）

*   rnn\_new: 本次迭代新加的反向邻居
    
*   rnn\_old: 原有 knng 中的反向邻居
    
```c
    nn_new = nn_new + rnn_new
    nn_old = nn_old + rnn_old
    all = nn_new + nn_old
      
    all_dists=parallel_compute_all_distances(nn_new, all)
    
    
    void parallel_try_insert_batch(u, candidates, dists) {
      u.lock()
      for v in candidates {
         u.UpdateKnnList(v, dists[i++])
         // 
      }
    }
    
    for u in nn_new {
      parallel_try_insert_batch(u, all, all_dists[u])
    }
    
    for u in nn_old {
      parallel_try_insert_batch(u, nn_new, transpose(all_dists)[u])
    }
```
join 阶段整体的复杂度分析：

*   计算每个点与候补邻居点的距离 all\_dists：O( ITER\* N \* (L+R)^^2^  \* Dimension）
    
*   将候补邻居插入到每个点的邻居列表中 parallel\_try\_insert\_batch：O( ITER\* N \* (L+R)^^2^ \* L）
    

其中:

*   ITER为迭代次数，当前规模下为7和8
    
*   N：节点数，规模1千万
    
*   Dimension： 200
    
*   L：邻居表的长度，范围为 150到200
    
*   R：反向邻居表的长度，范围为300到600
    

#### 距离计算的优化

即 parallel\_compute\_all\_distances，计算vector之间两两距离，这里是此流程中耗时最久的部分；

(𝐗 − 𝐘)^2^\= 𝐗^2^− 2𝐗𝐘 + 𝐘^2^

对于每个点，复杂度：(L+R)^^2^  \* Dimension

优化思路：

1.  eigen + intel mkl 矩阵计算优化
    
2.  更优的eigen写法
    
```
    // D 保存了na个点到nb个点的欧式距离结果
    // 维度：nb * na
    
    Eigen::MatrixXf A = nodes(Eigen::all, idA); // (200, na)
    Eigen::MatrixXf B = nodes(Eigen::all, idB).transpose(); // (nb, 200)
    // get square sum
    Eigen::MatrixXf A2 = square_sums(idA, Eigen::all).transpose();  // (1, na)
    Eigen::MatrixXf B2 = square_sums(idB, Eigen::all);  // (nb, 1)
    
    D.noalias() = -2 * B * A;
    D.noalias() += B2 * Eigen::MatrixXf::Ones(1, A2.cols());
    D.noalias() += Eigen::MatrixXf::Ones(B2.rows(),1) * (A2);
```
**优化点：**

1.  减少\* Eigen::MatrixXf::Ones()
    
2.  减少2因子
    
```
    D.colwise() = square_sums(idB);
    D.rowwise() += square_sums(idA).transpose();
    D.noalias() -=  nodes(Eigen::all, idB).transpose() * nodes(Eigen::all, idA);
```

#### 邻居表插入parallel\_try\_insert\_batch的优化

单个点的local join:

*   二分查找到要插入的位置 search：O( (L+R)^^2^ \* log(L))
    
*   std::vector 插入操作要逐个移动 insert: O( (L+R)^^2^ \* L)
    

**优化思路：**复杂度非常高，即使进行小的优化也可以带来显著的改善效果。
```
    插入数据共7199725626, 7e9=70亿
    移动次数共676746237269, 6e11
```

根据邻居表空间是否已经达到超参预设的大小，邻居表空间状态分两个阶段：

1.  not full的阶段：发生第一次迭代的前小半段
    
2.  full阶段：第一次迭代的后大半段，以及其余迭代
    

实战中，减少full阶段的不必要的比较和插入，将parallel\_try\_insert\_batch拆除两个函数：

1.  parallel\_try\_insert\_batch\_notfull
    
2.  parallel\_try\_insert\_batch\_full
    
```
    void parallel_try_insert_batch_multistage(u, candidates, dists) {
      if u.isfull() {
         parallel_try_insert_batch_full(u, candidates, dists)
      } else {
         parallel_try_insert_batch_notfull(u, candidates, dists)
      }
    }
    
    for u in nn_new {
      parallel_try_insert_batch_multistage(u, all, all_dists[u])
    }
    
    for u in nn_old {
      parallel_try_insert_batch_multistage(u, nn_new, transpose(all_dists)[u])
    }
```
update阶段的优化

update阶段整体的复杂度分析：

*   O( ITER \* N \* L \* （L+R））
    

其中:

*   ITER为迭代次数
    
*   N：节点数，规模1千万
    
*   L：邻居表的长度
    
*   R：反向邻居表的长度
    

**邻居表优化**

**复用vector，减少内存分配**：

1.  mergernn\_new nn\_new into nn\_new
    
2.  mergernn\_old nn\_old into nn\_old
    

原有的loop拆分为两个loop：

1.  第一个loop：加锁加入反向邻居
    
2.  第二个loop：加入正向邻居
    

初步效果：

1.  内存更少：少了约15G
    
2.  更快：内存分配次数更少，意味着更快
    

进一步优化：

nn\_new/ nn\_oldvector生命周期原先因为存在oom问题，需要在每次join后销毁。

本次优化加上其他内存优化后，可持续到整个图构造周期，而不至于oom。

效果：减少每次迭代的nn\_new/ nn\_old的内存重新分配，从而提升了处理时间。

生成knn图阶段的优化

**邻居去重**

邻居表存在重复邻居项，最终结果需要去重。

同时优化join阶段的UpdateKnnList，去除处理重复项的逻辑，统一放到最后去重。

```
    float pre_dist = -1, dist;
    unordered_set<uint32_t> mp;
    
    uint32_t pre_id = -1;
    for (uint32_t i = 0, k=0; k < K; ++i) {
            uint32_t id = pool[i].id;
            dist = pool[i].dist;
            if (dist > pre_dist || mp.find(id) == mp.end()){
    
                    if (dist > pre_dist) {
                          pre_dist = dist + 1e-4;
                          mp = {};
                    }
    
                    mp.insert(id);
                    data[k++]=id;
            } // else discard
    }
    
```

why?

由于eigen 向量化计算sum(X\*Y), sum的顺序不一致，会有float精度损失，导致每次计算同一对<u, v> 的距离时可能会有不同值，导致邻居表存在重复项。

复用内存，减少knn图内存开销

复用初始化时用来读取输入文件的矩阵，把后续构建的 knng 都保存到这个矩阵里，最终输出结果文件时也用了同一个矩阵对象，减少了 knng 图和输出文件的内存开销。

**参数调优**

主要是三个参数（[参数列表](https://github.com/aaalgo/kgraph/blob/master/doc/params.md)）：

ITER: 迭代次数

L: 邻居表的长度

R: 反向邻居表的长度

```
    for ITER in 7 8; do
      for L in {180..231..5}; do
        for R in {350..600..50}; do
          do_run || timeout_break
        done
      done
    done
```

最终提交参数：

ITER=8

L=186

R=550

## 参考文献

1.  Dong W, Moses C, Li K. Efficient k-nearest neighbor graph construction for generic similarity measures\[C\]//Proceedings of the 20th international conference on World wide web. 2011: 577-586.
    
2.  [https://github.com/for0nething/SIGMOD-Programming-Contest-2023.git](https://github.com/for0nething/SIGMOD-Programming-Contest-2023.git)
    

## 附录：

我们基于[https://github.com/for0nething/SIGMOD-Programming-Contest-2023.git](https://github.com/for0nething/SIGMOD-Programming-Contest-2023.git)（sigmod 2023冠军方案， 其基于开源的[kgraph](https://github.com/aaalgo/kgraph)），冠军方案的贡献点：

1.  加速欧式距离的计算
    

(𝐗 − 𝐘)^2^\= 𝐗^2^− 2𝐗𝐘 + 𝐘^2^

*   𝐗^2^： 每个点 precompute
    
*   𝐗𝐘: 矩阵相乘计算
    
*   矩阵计算通过Eigen库实现，同时利用Intel MK 加速计算
    

2.  锁优化
    
```
    
    for 𝐮 in neighbors:
      for 𝐯 in neighbors:
      Dist(u,v)
      Get_lock_and_update(u)
      Get_lock_and_update(v)
```    

问题：

*   频繁地获取的释放锁
    
*   空间局部性缓存利用较差，增加了 IO 负担
    
```
    Compute_all_dist(neighbors)// Vectorization
    for 𝐮 in neighbors:
      Get_lock(𝐮)
      Update_all_neighbors(𝐮)
```   

优化:

*   Less lock acquisition and release 减少了锁的获取和释放
    
*   有更好的空间局部性
    

3.  邻居结构体的内存优化
    
```
    struct Neighbor {
        uint32_t id;
        float dist;
        flag bool;
    }
```

\=》 
```
    struct Neighbor {
        uint32_t id; // id << 1 | flag
        float dist;
    }
```