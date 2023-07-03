def str_path_util(input_path: str):
    output_path = input_path.translate({ord(i): None for i in '$#[]/.?-'})
    return output_path
