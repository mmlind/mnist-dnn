

/**
 * @brief DEBUGGING function to validate (dereference pointers) and print an initialized network
 * nn Pointer to a neural network
 */


/*


void displayNetworkAdressesForDebugging(Network *nn){
    
    
    
    // alte version vom 24. funktioniert
    // neue nicht
    // versuche beide version zu vergleichen
    // in der neuen version ist die node groesse anders, weil mehr connections (+forward connections)
    
    
    
    // displaying node column-node structure
    for (int l=0;l<nn->layerCount;l++){
        
        Layer *layer = getNetworkLayer(nn, l);
        printf("\n");
        
        for (int c=0; c<layer->columnCount; c++){
            
            Column *column = getLayerColumn(layer, c);
            printf("\n");
            
            for (int n=0; n<column->nodeCount; n++){
                
                Node *node = getColumnNode(column, n);
                
                printf("Layer:%d    Column:%3d    Node:%'d = %p   backwardConn:%d   \n", l,c,n,node,node->backwardConnCount);
                
                //                int maxConn = (node->backwardConnCount==0) ? 0 : 5;
                int maxConn = node->backwardConnCount;
                
                for (int o=0; o<maxConn;o++){
                    
                    Connection *conn = &node->connections[o];
                    
                    printf("Layer:%d    Column:%3d    Node:%d = %p   Conn:%3d    = %p    TargetNode:%p    TargetWeight:%p  TargetWeight:%'13.10f \n", l,c,n,node,o,conn,conn->nodePtr,conn->weightPtr,*conn->weightPtr);
                    
                }
                
            }
            
        }
        
    }
    
    
    
    // displaying the weight block after the column/nodes
    for (int l=0;l<nn->layerCount;l++){
        
        Layer *layer = getNetworkLayer(nn, l);
        Weight *weight = layer->weightsPtr;
        
        printf("Layer:%d    Weights:%'d \n",l,(int)weight);
        
    }
    
    
}


*/