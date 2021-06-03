import os
import sys
import pandas as pd

_PATH = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.abspath(os.path.join(_PATH, os.path.pardir)))

from  machine_teacher.Reports import create_reports_from_configuration_folder

_CONFIGURATION_BASE_FOLDER = os.path.join(_PATH, "configs")
_CONFIGURATION_BASE_FOLDER = os.path.abspath(_CONFIGURATION_BASE_FOLDER)
_DEST_FOLDER = os.path.join(_PATH, "results")
_DEST_FOLDER = os.path.abspath(_DEST_FOLDER)

_ERROR_MSG = "should be 'python main_from_folder.py <configuration_folder_name>'"

_LOG_SHEET_NAME = 'log'

def main(conf_folder_name):
	conf_folder_path = os.path.join(_CONFIGURATION_BASE_FOLDER,
		conf_folder_name)

	res_folder_path, TRs = create_reports_from_configuration_folder(conf_folder_path,
		_DEST_FOLDER, True)

	# convert summary to excel
	summary_file_name = _get_summary_file_name(res_folder_path)
	summary_file_path = os.path.join(res_folder_path, summary_file_name)
	excel_summary_file_path = summary_file_path.replace(".csv", ".xlsx")
	
	# add TRs log to excel, in a new sheet ('log')
	df_TRs = _convert_TRs_to_dataframe(TRs)
	with pd.ExcelWriter(excel_summary_file_path,
		engine='openpyxl', mode='w') as writer: 
		df_TRs.to_excel(writer, sheet_name = _LOG_SHEET_NAME, index = False)


def _get_summary_file_name(res_folder_path):
	sufix = os.path.basename(os.path.normpath(res_folder_path))[6:]
	summary_file_name = "reports_summary" + sufix + ".csv"
	return summary_file_name

def _convert_TRs_to_dataframe(TRs):
    assert len(TRs) > 0, "A lista de <teaching results> está vazia..."
    header = TRs[0].log[0]
    
    # check if headers are identical
    for TR in TRs:
        header_TR = TR.log[0]
        assert header == header_TR
    
    header = ("Teacher", "Learner", "Dataset", "Id") + header
    TRs_table = []
    
    for (id_TR, TR) in enumerate(TRs):
        TR_triple = (TR.main_infos.teacher_name,
                  TR.main_infos.learner_name,
                  TR.main_infos.dataset_name)
        
        prefix = TR_triple + (id_TR+1,)
                  
        for i in range(1, len(TR.log)): #skip header
            line = prefix + TR.log[i]
            TRs_table.append(line)
            
    df = pd.DataFrame(TRs_table, columns = header)
        
    return df

if __name__ == "__main__":
	if len(sys.argv) != 2:
		raise KeyError(_ERROR_MSG)

	conf_folder_name = sys.argv[1].strip()
	main(conf_folder_name)
	