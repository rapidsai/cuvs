# arm tests sporadically run into https://bugzilla.redhat.com/show_bug.cgi?id=1722181.
# This is a workaround to ensure that OpenMP gets the TLS that it needs.

import sklearn
