# import dendrogram_generator
# import HMI
# 
# if __name__ == main
# 
# do n times:
#   generate dendrogram using dendrogram_generator
#   do m times:
#       generate graph from dendrogram using dendrogram.generate_graph -> convert nx graph to igraph
#       run all selected divisive algorithms -> return a merges list
#       run all selected agglomorative algorithms (currently, there are none)
#   scoring function = HMI(dendrogram, )