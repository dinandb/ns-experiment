# import dendrogram_generator
# import HMI
# 
# if __name__ == main
# 
# do n times:
#   generate dendrogram using dendrogram_generator
#   denote by M the merges list corresponding to dendrogram
#   do m times: hoi 
#       generate graph from dendrogram using dendrogram.generate_graph -> convert nx graph to igraph
#       run all selected divisive algorithms -> return a merges list M'
#       run all selected agglomorative algorithms (currently, there are none)
#   
#       scoring function = HMI(M, M')