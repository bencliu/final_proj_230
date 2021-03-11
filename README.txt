Key to File Structure + Processes

Data: 

  These are the 3 files that are needed to run a NN. Before each run, these files need to be referenced in the NN code.
  1. aws_file_dict_vUpdate2.p
    - dictionary of the following form dict{image ids : aws file paths}
  2. partition_vUpdate2.p
    - dictionary of the following form dict{"train": list of train ids, "val": list of val ids, "test": list of test ids}
  3. processed_label_dict.p
  ` - dictionary of the following form dict{image ids: processed yield}



