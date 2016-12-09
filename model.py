# coding=utf8

'''
You should compelte this code.
I suggest you to write your email and mobile phone number here
#############  Contact information   ##############
email: 
phone:
############  Contact information end  ############

'''

###############   import start    #################



##############     import end     #################


def predict(input_file_path, output_file_path):
    '''
    This function should load the test file, 
    use the model to predict all correct triples in each sentence, 
    and write the result to the output file.

    input_file_path is the path to the test data json file, input file format:
    [
        {
            "sentenceId": "eadf4c4d7eaa6cb767fa6d8c02555f5-eb85e9fb6ec57b2dd9ba53a8cc4b1625b18",
            "indexes": [0, 6, 0, 6, 0, 7, 13, 104, 146, 33, 1, 11, 8, 2, 6, 2, 9, 2, 7, 11, 14, 17, 18, 1, 12, 2, 6, 2, 9, 2, 10 ],  
            "times": [0, 2, 4 ],  
            "attributes": [23, 10 ],
            "values": [13, 15, 17, 25, 27, 29 ],
        }, 
        { ... }, 
        ...
    ]

    output_file_path is the path to the output json file, output file format:
    [
        {
            "sentenceId": "eadf4c4d7eaa6cb767fa6d8c02555f5-eb85e9fb6ec57b2dd9ba53a8cc4b1625b18",
            "results": [
                [0, 10, 13 ],
                [4, 10, 17 ], 
                [2, 10, 15 ]],
        },
        { ... }, 
        ...
    ]
    '''

    ############ Complete the code below ############

    json.dump(results, open(output_file_path, 'w'))
