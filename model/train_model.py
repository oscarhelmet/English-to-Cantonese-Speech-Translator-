import data_analysis as da
import data_preprocessing as dp 
import data_visualization as dv

train_df, test_df = dp.data_preprocessing()
losses = da.train(train_df, test_df)
dv.plot_image(losses)

