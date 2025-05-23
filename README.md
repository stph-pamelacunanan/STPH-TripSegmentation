# STPH-TripSegmentation

## Motivation
Unlike the STPH mobile app, the OBD device is unable to identify individual vehicle trips, necessitating the development of a trip segmentation method for more accurate telematics analysis. Additionally, the trip_id generated by the STPH mobile app may be unreliable, as signal interruptions or GPS issues can cause the mobile app to register a new trip even when the vehicle remains on the same journey.


## About the code: Trip segmentation and Trip super summaries

The Python notebook `Trip Summaries and Segmentation Library.ipynb` **creates a local library named `stph_trips`** that can be imported thereafter. The library is currently composed of four modules:

1. `.trip_segmentor` - houses the functions used for trip segmentation. The method leverages both temporal and spatial features that are common to both the STPH mobile app and telematics datasets. A detailed write up of the process behind the method can be accessed in this [document](https://docs.google.com/document/d/1Wx1Gbu541PiNX9VnlU_9R16H73s0uZJAOQIxGo5zjI8/edit?tab=t.0#heading=h.fmctsk7q3oz8).
2. `.trip_summary_telematics` – houses the functions used to produce the trip super summaries of the telematics dataset
3. `.trip_summary_STPHapp` – houses the functions used to produce the trip super summaries of the STPH mobile app dataset
4. `.trip_summary_STPH_AppTrips` – houses the functions used to produce the trip super summaries of the STPH mobile app-generated trips. This was created so that the trips generated by the app can be compared against the trips segmented by the algorithm in `trip_segmentor` module.

Lastly, the two other Python notebooks `Trip Segmentation - Telematics.ipynb` and `Trip Segmentation - STPH App.ipynb` demonstrates the use of the said library.

