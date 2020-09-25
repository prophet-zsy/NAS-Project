# NAS-deliver2

### 环境依赖
- python 3.5-3.7，numpy 1.16-1.18   
所有文件依赖
- tensorflow 1.13-1.14   
先验和评估依赖，当不使用先验加速，且自定义应用场景时，则不需要此依赖

### 搜索启动
1. `mkdir data` 创建 data文件夹
2. 将目标数据集移动到data目录下，或`ln -s` 建立软连接
3. 调整并确认nas_config.json中的参数符合自己的搜索需求
4. `python nas.py`启动搜索

### 配置文件参数说明（nas_config.json）
配置文件修改较频繁的参数含义如下：   
<table>
	<tr>
	    <th>模块</th>   
	    <th>参数名</th>   
	    <th>含义</th>   
	    <th>默认值</th>   
	    <th>备注</th>   
	</tr >
	<tr >
	    <td rowspan="11">nas_main</td>
	    <td>num_gpu</td>
	    <td>可用gpu个数</td>
	    <td>2</td>
	    <td>根据实际机器上的gpu个数来填写</td>
	</tr>
	<tr>
	    <td>block_num</td>
	    <td>需要搜索block个数</td>
	    <td>4</td>
	    <td>结合repeat_num一起控制网络深度，带pooling的情况下，block个数设置不应超过image_size/2</td>
	</tr>
	<tr>
	    <td>num_opt_best</td>
	    <td>train_winner阶段的采样次数</td>
	    <td>20</td>
	    <td>即最优拓扑结构诞生后，在其基础上继续迭代优化节点的次数</td>
	</tr>
	<tr>
	    <td>spl_network_round</td>
	    <td>每轮每个网络的采样次数</td>
	    <td>5</td>
	    <td>在竞赛阶段，每个拓扑结构的表现应该用几次评估表示，搜索时长会随着该值的增大而增大</td>
	</tr>
	<tr><td>max_player_nums</td>
	    <td>经过初筛后参与竞赛的网络最大个数</td>
	    <td>32</td>
	    <td>如果经验模块参与搜索，会讲枚举出的拓扑结构个数删减到该值</td>
	</tr>
	<tr>
	    <td>add_data_per_round</td>
	    <td>每轮添加数据量</td>
	    <td>1000</td>
	    <td>如果使用线性添加数据的方法，则该值起作用</td>
	</tr>
	<tr>
	    <td>add_data_for_confirm_train</td>
	    <td>每个block的confim阶段需要数据量</td>
	    <td>-1</td>
	    <td>调试时置100，缩短调试时间，搜索时置-1，添加全量数据</td>
	</tr>
	<tr>
	    <td >repeat_num</td>
	    <td>每个block搜得结果的堆叠次数</td>
	    <td>1</td>
	    <td>如果要加深网络，可以调节此值，将搜得结构堆叠</td>
	</tr>
	<tr>
	    <td >pred_mask</td>
	    <td>不使用预测模块</td>
	    <td>0</td>
	    <td>0表示使用，1表示不使用</td>
	</tr>
	<tr>
	    <td >filter_mask</td>
	    <td >不使用拓扑结构初筛模块</td>
	    <td >0</td>
	    <td >0表示使用，1表示不使用</td>
	</tr>
	<tr>
	    <td >link_node</td>
	    <td >每个block末尾是否添加pooling</td>
	    <td >true</td>
	    <td >true表示添加，一般用于分类，false表示不添加，一般用于去噪</td>
	</tr>
	<tr>
        <td rowspan="3">enum</td>
	    <td >depth</td>
	    <td >每个block主链深度</td>
	    <td >4</td>
	    <td >不包含末尾添加的pooling</td>
	</tr>
	<tr>
	    <td >width</td>
	    <td >支链最大个数 </td>
	    <td >2</td>
	    <td >枚举模块会在小于等于该值的范围内枚举</td>
	</tr>
    <tr>
	    <td >max_depth</td>
	    <td >支链最大深度</td>
	    <td >3</td>
	    <td >枚举模块会在小于等于该值的范围内枚举</td>
	</tr>
    <tr>
        <td rowspan="3">eva</td>
	    <td >task_name</td>
	    <td >搜索的实际场景</td>
	    <td >cifar-10</td>
	    <td >主控和评估会根据任务设定读取对应的数据集，添加新场景需要在主控导入模块部分定义新场景名称，然后在此设定即可</td>
	</tr>
	<tr>
	    <td >search_epoch</td>
	    <td >竞赛阶段的epoch设定</td>
	    <td >10</td>
	    <td >值越大，搜索越精确，用时越大</td>
	</tr>
    <tr>
	    <td >confirm_epoch</td>
	    <td >confim阶段的epoch设定</td>
	    <td >100</td>
	    <td >confirm阶段的训练结果用作下一个block的base，需要在充分训练和耗时之间平衡</td>
	</tr>
    <tr>
        <td rowspan="3">spl</td>
	    <td >skip_max_dist</td>
	    <td >跨层连接最大距离</td>
	    <td >3</td>
	    <td >一般随主链长变化而变化</td>
	</tr>
	<tr>
	    <td >skip_max_num</td>
	    <td >跨层连接最大个数</td>
	    <td >2</td>
	    <td >采样模块在不超过此值之间采样</td>
	</tr>
    <tr>
	    <td >space</td>
	    <td >节点操作的搜索空间</td>
	    <td >默认值详见配置文件</td>
	    <td >默认在已有操作上搜索，如需添加新操作，可按已有格式添加</td>
	</tr>
</table>

### 已包含应用场景
- 分类（evaluator_classificatoin.py）
1. cifar10
2. cifar100
3. tiny-imagenet
- 去噪
1. waterloo,(evaluator_denoise.py)
2. SIDD,(evaluator_SIDD.py)

### 添加新的应用场景

1. 定义搜索场景。主搜索框架与评估的交互界面如下：主控传给评估task_item(EvaScheduleItem实例化的对象，一个实例对应一个任务，内部包含所有该任务的信息)，评估返回给主控评分（将评分填在task_item中返回），因此如需添加新的应用场景，将其定义为新的评估模块即可,接口信息如下：
```
·主控与评估的接口函数：
    score = Evaluator().evaluate(task_item)
·task_item由主控赋值评估读取的属性：
    nn_id：当前拓扑结构的编号
    network_item：当前block的结构
    pre_block：所有先前block的结构
    epoch：本次评估需要设定的epoch
    data_size：本次评估需要添加的数据量
·task_item由评估赋值主控读取的属性：
    model_params：本次评估模型的大小
```
2. 按照模板evaluator_user.py添加适合于自己特定应用场景的内容, 重命名为evaluator_***.py的格式
3. 在nas.py加入导入新应用场景评估文件的逻辑，将nas_config.json中task_name设置为新应用场景
4. 调整搜索配置参数及搜索空间
5. `python nas.py` 启动搜索

#### 经验加速模块的使用
- 经验加速模块分两部分：1、拓扑结构和节点操作的关系作为先验，2、拓扑结构之间的质量比较作为先验（该功能暂未完善，预计下周更新）。使用经验加速模块，可以缩短搜索时长。是否使用经验加速模块，可通过配置文件中的pred_mask和filter_mask两个变量来设置
- 经验模块的代码针对所有应用场景是通用的，但经验模型不是通用的，如添加新的应用场景需要对经验模块的模型进行更新训练（经验模块已经适配代码中已包含的应用场景）。目前已开放预测节点操作模块的训练接口函数，网络质量预测模块接口暂未完善，待更新。
```
·预测节点操作模块的预测接口函数:
   Predictor.predictor(pre_block, graph_full)  
   pre_block:list[graph_full...],要预测的当前block的所有前置block  
   graph_full:list[[]], 当前需要预测的block的拓扑结构  
   
·预测节点操作模块的训练接口函数：  
   Predictor.train_model(graph_group, cell_list_group)  
   graph_group:训练集的输入数据。注：每条数据是一个完整的网络拓扑结构，而非block的集合  
   cell_list_group:训练集的标签数据。每条数据为网络拓扑结构对应的操作，如卷积，池化  

·网络质量预测模块预测接口函数：
   待完善
   
·网络质量预测模块训练接口函数：
   待完善
```
##### graph_full说明
graph_full是项目中用于表示网络拓扑结构的数据结构。graph_full用邻接表来表示图的拓扑结构，这里关键的是对节点的编码方式，首先确定主链，主链是从输入节点到输出节点的最长路径，先对主链进行编码，再对各支链编码。如：[[1,2],[2,5],[3],[4],[],[4]]表示主链长度为5，节点0连接到节点1和2，节点1连接到节点2和5，这里编号为5的节点表示支链上的节点，由于节点4，即输出节点没有下一个节点，所以为空。  
##### cell_list说明
cell_list是类Cell的集合，类Cell定义在模块base中，表示节点操作的数据结构。  
##### 训练数据的特征提取
提取的特征如下：节点数目，链数目，最长链的长度，编号，最短链的长度，编号。所有链长度的期望和方差。是否端点的标记，编号，连接端点的支链的数目。所有连接段端点的支链长度的期望和方差。端点所在支链的最长长度和最短长度，节点在所在链中的相对位置以及该链在网络结构中出现的概率，支链的长度，支链所在端点的编号，与节点所在支链的其他支链中的最长长度和最短长度以及所有这些支链的期望和方差  


