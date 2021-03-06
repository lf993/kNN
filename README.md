# kNN
## R语言算法：kNN(临近分类器)
<p>kNN的利用的是距离最近且多的就归为一类的思想，给定一个数据集，里面的数据是贴有标签的，然后输入你的测试集，选择k的大小，一般为数据集的算数平方根为k值大小，但是k要取整数，然后用测试集里的数据和你的数据求欧式距离（坐标差的平方和开根号），选取k个最小的距离为参照物，看其所带的标签，数量多者为你的测试集的类型。</p>
<p>数据的标准化就是为了防止数值大的对数值小的影响</p>
当然它也有缺陷，不足如下：

1. 不产生模型，在发现特征之间关系上的能力有限
2. 分类阶段很慢
3. 需要大量的内存
4. 名义变量（特征）和缺失数据需要额外处理

>此代码来源于《机器学习与R语言》

如果不想下载数据集的csv格式，可以参考赋值给url那一行开始到最后的部分，可以直接从网络上读取<br>
k值不是固定不变的，而是需要通过评估的值进行调整，这是试出来，算数平方根不一定是最优解。<br>
因为原数据发生了变化，所以它与书本上的结果不一样，所以你要相信自己的结果。
