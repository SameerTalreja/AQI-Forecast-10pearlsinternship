import hopsworks

project = hopsworks.login(api_key_value="TPEXjaYs3fyoakXd.DPFtG3oOvgwovusWuHWSNBCTBjn852uFv6TgKczn89nfTI3913hjzmSk7JJ2L2Zv")
print("Connected to project:", project.name)