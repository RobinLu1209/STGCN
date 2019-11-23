'''
第一阶段称为Modularity Optimization，主要是将每个节点划分到与其邻接的节点所在的社区中，以使得模块度的值不断变大；
第二阶段称为Community Aggregation，主要是将第一步划分出来的社区聚合成为一个点，即根据上一步生成的社区结构重新构造网络。
重复以上的过程，直到网络中的结构不再改变为止。
因为要带权重,所以读数据才那么慢？
'''
import numpy as np
import pandas as pd

class FastUnfolding:
    '''
        从一个csv文件路径中创建一个图.
        path: 文件中包含 "node_from node_to" edges (一对一行)
    '''
    #classmethod 修饰符对应的函数不需要实例化，不需要 self 参数，
    # 但第一个参数需要是表示自身类的 cls 参数，可以来调用类的属性，类的方法，实例化对象等

    @classmethod
    def from_csv(cls, data):
        dataarray = np.asarray(data)
        nodes = {}
        edges = []
        for n in dataarray:
            nodes[n[0]] = 1
            nodes[n[1]] = 1
            w = 1
            if len(n) == 3:
                w = int(n[2])
            edges.append(((n[0], n[1]), w))
        # 用连续的点重编码图中的点
        nodes_, edges_,d = in_order(nodes, edges)
        return cls(nodes_, edges_), d

    '''
        从一个txt文件路径中创建一个图.
        path: 文件中包含 "node_from node_to" edges (一对一行)
    '''
    @classmethod
    def from_file(cls, path):

        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        nodes = {}
        edges = []
        for line in lines:
            n = line.split()
            if not n:
                break
            nodes[n[0]] = 1#生成一个字典，记录原始图中出现的点
            nodes[n[1]] = 1
            w = 1
            if len(n) == 3:#有权重是权重，没权重是1
                w = int(n[2])
            edges.append(((n[0], n[1]), w))
        # 用连续的点重编码图中的点
        nodes_, edges_,d = in_order(nodes, edges)
        print("%d nodes, %d edges" % (len(nodes_), len(edges_)))
        return cls(nodes_, edges_),d


    '''
        从一个gml文件路径中创建一个图.

    '''

    @classmethod
    def from_gml_file(cls, path):
        f = open(path, 'r')
        lines = f.readlines()
        f.close()
        nodes = {}
        edges = []
        current_edge = (-1, -1, 1)
        # dic ={}
        in_edge = 0
        for line in lines:
            words = line.split()
            if not words:
                break
            if words[0] == 'id':
                # a = int(words[1])
                nodes[int(words[1])] = 1
            # if words[0] == 'label':
            #     dic[words[1]] = a
            elif words[0] == 'source':#当读到source的时候，开始刷新current_edge
                in_edge = 1
                current_edge = (int(words[1]), current_edge[1], current_edge[2])
            elif words[0] == 'target' and in_edge:
                current_edge = (current_edge[0], int(words[1]), current_edge[2])
            elif words[0] == 'weight' and in_edge:
                current_edge = (current_edge[0], current_edge[1], int(words[1]))
            elif words[0] == ']' and in_edge:
                edges.append(((current_edge[0], current_edge[1]),current_edge[2]))
                current_edge = (-1, -1, 1)
                in_edge = 0#读完一个边，添加到edges中，并刷新current_edge和in_edge
        nodes, edges,d = in_order(nodes, edges)
        print("%d nodes, %d edges" % (len(nodes), len(edges)))
        return cls(nodes, edges), d

    '''
        初始化方法.
        nodes: 一列整数列表
        edges: a list of ((int, int), weight) pairs
    '''
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        #  m 网络中所有边的权重之和
        #  k_i 与点i相连的边的权重总和
        self.m = 0
        self.k_i = [0 for n in nodes]
        self.edges_of_node = {}
        self.w = [0 for n in nodes]

        for e in edges:
            self.m += e[1]
            self.k_i[e[0][0]] += e[1]
            self.k_i[e[0][1]] += e[1] # 一开始没有环
            
            # 在edges_of_node中保存该点出现的所有边，有环的时候只用保存一次
            if e[0][0] not in self.edges_of_node:
                self.edges_of_node[e[0][0]] = [e]
            else:
                self.edges_of_node[e[0][0]].append(e)

            if e[0][1] not in self.edges_of_node:
                self.edges_of_node[e[0][1]] = [e]
            elif e[0][0] != e[0][1]:
                self.edges_of_node[e[0][1]].append(e)
        # access community of a node in O(1) time
        # O（1）时间节点的访问社区
        self.communities = [n for n in nodes]#这时所有的点是一个社区
        self.actual_partition = []


    '''
        应用Fast Unfolding方法
    '''
    def apply_method(self):
        network = (self.nodes, self.edges)
        best_partition = [[node] for node in network[0]]
        best_q = -1
        i = 1
        process = []
        partitions ={}
        while 1:
            i += 1
            partition = self.first_phase(network)#第一步，把点聚合
            q = self.compute_modularity(partition)#计算现下分区的modularity
            partition = [c for c in partition if c]#把partition中为空的列表删除
            # clustering initial nodes with partition
            if self.actual_partition:
                actual = []
                for p in partition:
                    part = []
                    for n in p:
                        part.extend(self.actual_partition[n])
                    actual.append(part)
                self.actual_partition = actual
            else:
                self.actual_partition = partition
            if q == best_q: # 如果本轮迭代modularity没有改变，则认为收敛，停止
                break

            network = self.second_phase(network, partition)
            best_partition = partition
            best_q = q
            a = len(self.actual_partition)
            partitions[a] = self.actual_partition
            process.append((a,best_q))
        #print(process)
        print("%d community" % len(self.actual_partition))
        return (self.actual_partition, best_q, process, partitions)


    '''
        计算当下网络的modularity
        partition: a list of lists of nodes
    '''
    def compute_modularity(self, partition):
        q = 0
        m2 = self.m * 2#self.m是全图中的总权重
        '''
        self.s_in是该社区内部边数的两倍，所以这里分母要乘以2
        self.s_tot是该社区内部边数的两倍加上与外部社区边的数量'''
        for i in range(len(partition)):
            q += self.s_in[i] / m2 - (self.s_tot[i] / m2) ** 2
        return q

    '''
        计算node再社区c时的modulari增益.
        node: an int  点
        c: an int  社区
        k_i_in: the sum of the weights of the links from _node to nodes in _c  node与社区c中的点边的权重总和
    '''
    def compute_modularity_gain(self, node, c, k_i_in):
        return 2 * k_i_in - self.s_tot[c] * self.k_i[node] / self.m

    '''
        fast unfolding 方法的第一步.
        _network: a (nodes, edges) pair
    '''
    def first_phase(self, network):
        # 创建初始化社区
        best_partition = self.make_initial_partition(network)
        while 1:
            improvement = 0
            for node in network[0]:
                node_community = self.communities[node]#每个点的所属社区号
                # 默认最佳社区是它自己，当移动到相邻社区中时modularity不变，再移回该社区
                best_community = node_community
                best_gain = 0
                # 从点原始所在的社区中把这个点删除
                best_partition[node_community].remove(node)
                best_shared_links = 0
                #在best_shared_linnks中保存与之有关的边的权重
                for e in self.edges_of_node[node]:
                    if e[0][0] == e[0][1]:
                        continue
                    #因为可能是一个有向图，所以判断条件才那么多！！
                    if e[0][0] == node and self.communities[e[0][1]] == node_community or e[0][1] == node and self.communities[e[0][0]] == node_community:
                        best_shared_links += e[1]
                self.s_in[node_community] -= 2 * (best_shared_links + self.w[node])#node本来所在的社区，s_in减小
                self.s_tot[node_community] -= self.k_i[node]#node所在的社区，s_tot减少
                self.communities[node] = -1 #这时node不属于任何社区，第node个数赋值为-1
                communities = {} # 只考虑不同社区中的邻居，不同社区中可能有多个邻居，只需一次
                for neighbor in self.get_neighbors(node):
                    community = self.communities[neighbor]
                    if community in communities:
                        continue
                    communities[community] = 1
                    shared_links = 0
                    for e in self.edges_of_node[node]:
                        if e[0][0] == e[0][1]:
                            continue
                        if e[0][0] == node and self.communities[e[0][1]] == community or e[0][1] == node and self.communities[e[0][0]] == community:
                            shared_links += e[1]
                    # 计算移动点后，modularity的变化值
                    gain = self.compute_modularity_gain(node, community, shared_links)
                    if gain > best_gain:
                        best_community = community
                        best_gain = gain
                        best_shared_links = shared_links
                # 把该点移动到modularity变化正值最大的社区中
                best_partition[best_community].append(node)
                self.communities[node] = best_community#这时node属于该社区，第node个数赋值为该社区的编号
                self.s_in[best_community] += 2 * (best_shared_links + self.w[node])
                self.s_tot[best_community] += self.k_i[node]
                if node_community != best_community:
                    improvement = 1
            if not improvement:#直到没有提高，才跳出while循环,输出best_partition
                break
        return best_partition

    '''
        一个与node相邻的点的生成器.
        _node: an int
    '''
    def get_neighbors(self, node):#yield 的作用就是把一个函数变成一个 generator
        for e in self.edges_of_node[node]:
            if e[0][0] == e[0][1]: # 点与自己并不是邻居
                continue
            if e[0][0] == node:
                yield e[0][1]
            if e[0][1] == node:
                yield e[0][0]

    '''
        创建初始化社区（把环加到s_in里)
        network: a (nodes, edges) pair
    '''
    def make_initial_partition(self, network):
        partition = [[node] for node in network[0]]
        self.s_in = [0 for node in network[0]]
        self.s_tot = [self.k_i[node] for node in network[0]]
        for e in network[1]:
            if e[0][0] == e[0][1]: # only self-loops
                self.s_in[e[0][0]] += e[1]
                self.s_in[e[0][1]] += e[1]
        return partition

    '''
        Performs the second phase of the method.
        _network: a (nodes, edges) pair
        _partition: a list of lists of nodes
    '''
    def second_phase(self, network, partition):
        nodes_ = [i for i in range(len(partition))]
        # 重新编码社区编号，原先的可能是断断续续的，编码为连续从0开始
        communities_ = []
        d = {}
        i = 0
        for community in self.communities:
            if community in d:
                communities_.append(d[community])
            else:
                d[community] = i
                communities_.append(i)
                i += 1
        self.communities = communities_
        # building relabelled edges
        edges_ = {}
        for e in network[1]:
            ci = self.communities[e[0][0]]
            cj = self.communities[e[0][1]]
            try:
                edges_[(ci, cj)] += e[1]
            except KeyError:
                edges_[(ci, cj)] = e[1]
        edges_ = [(k, v) for k, v in edges_.items()]#由字典变为列表
        # 重新计算k_i and 按点保存边
        self.k_i = [0 for n in nodes_]
        self.edges_of_node = {}
        self.w = [0 for n in nodes_]
        for e in edges_:
            self.k_i[e[0][0]] += e[1]
            self.k_i[e[0][1]] += e[1]
            if e[0][0] == e[0][1]:#这里出现了环，初始时没有
                self.w[e[0][0]] += e[1]
            if e[0][0] not in self.edges_of_node:
                self.edges_of_node[e[0][0]] = [e]
            else:
                self.edges_of_node[e[0][0]].append(e)
            if e[0][1] not in self.edges_of_node:
                self.edges_of_node[e[0][1]] = [e]
            elif e[0][0] != e[0][1]:
                self.edges_of_node[e[0][1]].append(e)
        # resetting communities
        self.communities = [n for n in nodes_]
        return (nodes_, edges_)



'''
    创建一个连续点的图，把原来可能不连续的点编码为连续的点
    nodes_: 连续点的集合
    edges_: 列表，格式是 ((int, int), weight) 对
'''
def in_order(nodes, edges):
        # rebuild graph with successive identifiers
        nodes = list(nodes.keys())#把nodes字典中的键提取出来，这就是原始表中的点
        nodes.sort()#把nodes字典排序，以便后续的加点
        i = 0#新的点从0开始
        nodes_ = []
        d = {}#相当于标签，键是原始图中的点，值是新生成数据中的点
        for n in nodes:
            nodes_.append(i)
            d[n] = i
            i += 1
        edges_ = []
        for e in edges:
            edges_.append(((d[e[0][0]], d[e[0][1]]), e[1]))
        return (nodes_, edges_,d)


def main():
    print("=== Fast Unfolding Community Discovery Algorithm ===")
    network = pd.read_table('C:/Users/lu/Desktop/jupyter/network/test_network.txt', names=['start','end'], sep = " ")
    dataarray = np.asarray(network)
    nodes = {}
    edges = []
    for n in dataarray:
        nodes[n[0]] = 1
        nodes[n[1]] = 1
        w = 1
        if len(n) == 3:
            w = int(n[2])
        edges.append(((n[0], n[1]), w))
    # 用连续的点重编码图中的点
    nodes_, edges_,d = in_order(nodes, edges)
    test = FastUnfolding(nodes_, edges_)
    actual_partition, best_q, process, partitions = test.apply_method()
    print("actual_partition = ", actual_partition)
    print("best_q = ", best_q)
    print("process = ", process)
    print("partitions = ", partitions)


if __name__ == "__main__":
    main()

