def visualise_reconstruction(writer, originals, reconstructions):

    writer.add_images('originals', originals)
    writer.add_images('reconstructions', reconstructions)