import openeo

def connect():
    connection = openeo.connect("https://openeo.dataspace.copernicus.eu/openeo/1.2")
    connection.authenticate_oidc()
    return connection
