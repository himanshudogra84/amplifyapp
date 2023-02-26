import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config (layout="wide")

st.title('Machine Learning Portal')


def main():
    menu = ['Regression','Classification','Segmentation']
    choice = st.sidebar.selectbox('Choose Application',menu)
    st.header(choice)

    if choice == 'Regression':
        st.header('Dataset')
        data_file = st.sidebar.file_uploader('Upload CSV File',type=['csv'])
        
        if data_file is not None:
            global df, df_x, df_y
            
            st.text('')
            st.text('-----------------------------------------------------------------------------------------------------------------------------------------')
            st.text('')
            col1, col2 = st.columns([1,2])
            file_details = pd.DataFrame([data_file.name,data_file.type,data_file.size],index=['Filename','Filetype','Filesize'],columns=['Features'])
            
            with col1:
                st.write('File Properties:')
                st.dataframe(file_details)
                #.style.set_properties(**{'background-color': 'black','color': 'lawngreen','border-color': 'white'}))
            
            # Load DataFrame
            df = pd.read_csv(data_file)
            with col2:
                st.write('First 5 rows of Dataset:')
                st.dataframe(df.head())
            
            
            st.text('')
            st.text('-----------------------------------------------------------------------------------------------------------------------------------------')
            st.text('')

            # Data size
            col3, col4 = st.columns(2)
            null_values = pd.DataFrame(df.isnull().sum(),columns=['Missing Values'])
            data_type = pd.DataFrame(df.dtypes,columns=['Data Type'])
            with col3:
                st.write('Data Type and Missing Values:')
                st.dataframe(pd.concat([data_type,null_values],axis=1))
            with col4:
                st.write('Number of Rows & Columns in Uploaded File:', pd.DataFrame(df.shape,index=['Rows','Columns'],columns=['Size']))
                df = df.dropna(how='any',axis=0)
                st.text('')
                st.text('')
                st.write('Number of Rows & Columns after dropping missing values:', pd.DataFrame(df.shape,index=['Rows','Columns'],columns=['Size']))

            

            # Split x and y
            df_dummy = list(df.columns)
            df_dummy.insert(0, '--Select--',)
            try:
                select_y = st.sidebar.selectbox('Choose Y', options=df_dummy)
                df_x = df.drop(select_y, axis=1)
                df_y = pd.DataFrame(df[select_y])
            

                # Convert categorical columns to numeric
                df_x_cat = df_x.select_dtypes(include=['object']).apply(lambda x: pd.factorize(x)[0])
                df_x_num = df_x.select_dtypes(include=[np.number])
                df1_x = pd.concat([df_x_num,df_x_cat], axis=1)
                df1_x = df1_x[df_x.columns]
                try:
                    if df_y.dtypes[0] == 'object':
                        df_y = df_y.select_dtypes(include=['object']).apply(lambda x: pd.factorize(x)[0])
                except Exception as e:
                    print(e)
                df_new = pd.concat([df1_x,df_y],axis=1)

                # Create Plots of dataset
                df_y_0 = list(df_y.columns)
                df_y_0.insert(0,'Select Variable')
                df_x_num = list(df1_x.select_dtypes(['int','float']).columns)
                df_x_num.insert(0,'Select Variable')
                with open(data_file.name,'wb') as f:
                    f.write(data_file.getbuffer())
                st.success('File Saved')
                #chart_select = st.sidebar.selectbox(label='Select Visualization', options=['Scatter Plot','Line Plot'])

                st.text('')
                st.text('-----------------------------------------------------------------------------------------------------------------------------------------')
                st.text('')

                try:
                    st.header('Data Plots')
                    x_values = st.selectbox('X axis', options=df_x_num)
                    y_values = select_y
                    plot1 = px.scatter(data_frame=df,x=x_values,y=y_values, color_discrete_sequence=['cornflowerblue'])
                    st.subheader('Scatter Plot')
                    st.plotly_chart(plot1)
                    plot2 = px.histogram(data_frame=df,x=x_values, color_discrete_sequence=['burlywood'])
                    plot2.update_traces(marker_line_width=1,marker_line_color="white")
                    st.subheader('Histogram')
                    st.plotly_chart(plot2)
                except Exception as e:
                    print(e)
                
                st.text('')
                st.text('-----------------------------------------------------------------------------------------------------------------------------------------')
                st.text('')

                # Data Normalization
                st.header('Data Engineering')
                scaler = MinMaxScaler().fit(df_new)
                df_trans = pd.DataFrame(scaler.transform(df_new),columns=df_new.columns)
                st.write('First 5 rows after Numeric Transformation & Normalization')
                st.dataframe(df_trans.head().round(2))

                # Correlation Matrix
                st.write('Correlation Matrix')
                corrmatrix = df_trans.corr().style.background_gradient(cmap ='Blues').set_properties(**{'font-size': '20px'}).set_properties()
                st.dataframe(corrmatrix)

                # Split x and y of Normalized matrix
                df_trans_x = df_trans.drop(select_y, axis=1)
                df_trans_y = pd.DataFrame(df_trans[select_y])

                st.text('')
                st.text('-----------------------------------------------------------------------------------------------------------------------------------------')
                st.text('')

                # Split training and testing data and Fitting Linear Regression to the training set
                st.header('Data Modeling')
                x_train, x_test, y_train, y_test = train_test_split(df_trans_x, df_trans_y, test_size = 0.3, random_state = 0, shuffle=True)
                model = LinearRegression()
                model.fit(x_train, y_train)
                st.write('Model Coefficients:')
                coef = pd.DataFrame(model.coef_, columns=x_train.columns)
                st.dataframe(coef)
                st.write("Model Intercept:", '%.2f' % model.intercept_)

                # Generate Prediction on train and test set
                y_pred_train = model.predict(x_train)
                y_pred_test = model.predict(x_test)
                
                # Performance Metrics
                st.subheader('Performance Metrics:')
                performance = pd.DataFrame([[mean_squared_error(y_train, y_pred_train),r2_score(y_train, y_pred_train)],[mean_squared_error(y_test, y_pred_test),r2_score(y_test, y_pred_test)]],columns=['MSE','R2-score'], index=['Training set','Test set']).mul(100).round(2).astype(str).add(' %')
                st.table(performance)

                st.text('')
                st.text('-----------------------------------------------------------------------------------------------------------------------------------------')
                st.text('')

                if y_pred_test is not None:
                    test_file = st.sidebar.file_uploader('Upload File for Prediction', type=['csv'])
                    if test_file is not None:
                        st.header('Test Data')
                        col1, col2 = st.columns([1,2])
                        file_details = pd.DataFrame([data_file.name,data_file.type,data_file.size],index=['Filename','Filetype','Filesize'],columns=['Features'])
                        with col1:
                            st.write('File Properties:')
                            st.dataframe(file_details)
                
                        # Load DataFrame
                        df_test = pd.read_csv(test_file)
                        with col2:
                            st.write('First 5 rows of Dataset:')
                            st.dataframe(df_test.head())

                        st.text('')
                        st.text('-----------------------------------------------------------------------------------------------------------------------------------------')
                        st.text('')

                        # Data size
                        col3, col4 = st.columns(2)
                        null_test_values = pd.DataFrame(df_test.isnull().sum(),columns=['Missing Values'])
                        data_test_type = pd.DataFrame(df_test.dtypes,columns=['Data Type'])
                        with col3:
                            st.write('Number of missing values in each column:')
                            st.dataframe(pd.concat([data_test_type,null_test_values],axis=1))
                        with col4:
                            st.write('Number of Rows & Columns in Uploaded File:', pd.DataFrame(df_test.shape,index=['Columns','Rows'],columns=['Size']))
                            df_test_new = df_test.dropna(how='any',axis=0)
                            df_test_new = df_test_new.reset_index()
                            df_test_new = df_test_new.iloc[:,1:]
                            st.text('')
                            st.text('')
                            st.write('Number of Rows & Columns after dropping missing values:', pd.DataFrame(df_test_new.shape,index=['Columns','Rows'],columns=['Size']))

                        st.text('')
                        st.text('')
                        
                        # Training and Test Data Column placement
                        st.write("Mapping of Columns")
                        st.table(pd.DataFrame([df_x.columns,df_x.dtypes,df_test_new.columns,df_test.dtypes],index=['Training Data','Training Data type','Test Data','Test Data Type']).T)
                        df_test_cat = df_test_new.select_dtypes(include=['object']).apply(lambda x: pd.factorize(x)[0])
                        df_test_num = df_test_new.select_dtypes(include=[np.number])
                        df1_test = pd.concat([df_test_num,df_test_cat], axis=1)
                        df1_test = df1_test[df_test_new.columns]

                        st.text('')
                        st.text('-----------------------------------------------------------------------------------------------------------------------------------------')
                        st.text('')

                        st.header('Machine Learning Output')
                        scaler_test = MinMaxScaler().fit(df1_test)
                        df_test_trans = pd.DataFrame(scaler_test.transform(df1_test),columns=df1_test.columns)

                        test_prediction = model.predict(df_test_trans)
                        target_scale = pd.DataFrame(scaler.scale_,index=df.columns)[0][df.shape[1]-1]
                        output = pd.DataFrame((test_prediction/target_scale).round(2),columns=['Output'])
                        df_output = pd.concat([df_test_new,output],axis=1)
                        st.dataframe(df_output)

                        csv = df_output.to_csv().encode('utf-8')
                        st.download_button(label='Download Output File',data=csv,file_name='Output.csv',mime='csv')
            except Exception as e:
                print(e)
                    

            
            


    elif choice == 'Document Files':
        st.subheader('Document Files')

    else:
        st.subheader('About')

    
    


__name__ == '__main__'
main()
