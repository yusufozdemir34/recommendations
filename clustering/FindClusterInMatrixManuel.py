import numpy as np
import networkx as nx
import numpy as np
from collections import defaultdict


class Clustering:
    def findClusterInMatrix(aData, p_VeriSize):
        aData = np.asmatrix('14   2; 1     18; 2     16; 1    10; 14    11; 6    1;  1  4; 7     9; 8     9')
        cNumber = 1;
        ClusteredFlag = -1
        aDataTemp = aData

        L = np.size(aData, 0)
        rNumber = L(1);
        cData = np.zeros(rNumber, 1);
        cData[0] = 1

        for i in range(0, rNumber):
            if cData(i) == 0:
                cNumber = cNumber + 1
                cData[i] = cNumber

#             if aDataTemp[i, 1] != ClusteredFlag:
#                 rIndex = np.where(aDataTemp == aDataTemp[i, 1])
#                 [rIndex, cIndex] = find(aDataTemp == aDataTemp(i, 1))
#                 cData(rIndex(:))=cData(i);
#                 aDataTemp(sub2ind(L, rIndex, cIndex)) = ClusteredFlag;
#                 np.ravel_multi_index((1, 0, 1), dims=(3, 4, 2), order='F')
#                 13
#
#             if aDataTemp(i, 2)~=ClusteredFlag
#             [rIndex, cIndex] = find(aDataTemp == aDataTemp(i, 2));
#             cData(rIndex(:))=cData(i);
#             aDataTemp(sub2ind(L, rIndex, cIndex)) = ClusteredFlag;
#             end
#
#
# y = zeros(p_VeriSize, 1);
#
# for i=1:rNumber
# for j=1:2
# y(aData(i, j)) = cData(i);
# end
# end
#
# TotalClusterNumber = cNumber;
