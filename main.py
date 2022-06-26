from neural_network import train_data_images

def main():
    """
    Main function for running python script.
    """
    train_data_images("config_two_hidden.ini",
                      verbose=False,
                      show_num_images=5)

if __name__ == '__main__':
    main()