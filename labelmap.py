import pandas as pd

def create_label_map(excel_file,label_map_file, chunksize=1000):
    global next_id #declare the global variable
    next_id=1   #initialize next_id to 1

    with open(label_map_file,'w',encoding='utf-8') as f:
        for chunk in pd.read_csv(r'C:\Users\aujal\OneDrive\Desktop\python_OCR_application\dataset\medicine_names.csv', chunksize=chunksize):
            medicine_name= chunk.iloc[:,1].tolist()
            
            for name in medicine_name:
                #ensure the next_id doesn't exceeds the last index of the excel file
                if next_id>253973:
                    break
                ## Write the label item with the current next_id
                f.write(f'item {{\n')
                f.write(f'  id:{next_id}\n')
                f.write(f'  name: \'{name}\'\n')
                f.write(f'}}\n')

                next_id += 1 #Increment next_id for the next item


if __name__=='__main__':
    create_label_map(r'C:\Users\aujal\OneDrive\Desktop\python_OCR_application\dataset\medicine_names.csv',
                     r'C:\Users\aujal\OneDrive\Desktop\python_OCR_application\dataset\label_map.pbtxt',
                     chunksize=1000)