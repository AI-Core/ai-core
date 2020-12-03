def visualise_reconstruction(writer, originals, reconstructions, label):

    writer.add_images(f'originals/{label}', originals)
    writer.add_images(f'reconstructions/{label}', reconstructions)