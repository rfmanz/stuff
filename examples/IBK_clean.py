from pyutils import *
path = 'D:/Downloads/interbank20.zip'
read_data(path)

def rcc_train():
    rcc_train,y_train,productos = read_data(path, True,'dt',dataframes="rcc_train,y_train,productos")
    y_train = y_train.target*1
    rcc_train2 = rcc_train.copy()
    reduce_memory_usage(rcc_train2)
    rcc_train2.columns = rcc_train2.columns.str.lower()
    rcc_train2 = rcc_train2.merge(
        productos, how='left', left_on="producto", right_on="Productos")
    rcc_train2.rename(columns={"C0": "productos_nm"}, inplace=True)
    rcc_train2.drop(columns="Productos", inplace=True)
    rcc_train2.productos_nm.fillna("NULL",inplace=True)
    rcc_train2.drop(rcc_train2[rcc_train2.producto.astype(int).isin([36,41])].index, inplace=True)
    rcc_train2.codmes = rcc_train2.codmes.astype(str)

    return rcc_train2, y_train, productos, rcc_train

rcc_train2, y_train, productos,rcc_train = rcc_train()

