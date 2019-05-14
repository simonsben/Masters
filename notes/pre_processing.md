# Pre-processing

## Original pre-processing

Pre-processing is the transform applied to the data before it interacts with the classification model.
In Hannah's work the pre-processing applied was

* Make all characters lowercase
* Remove emojis
* Remove URLs
* Remove punctuation
* Remove numbers
* Remove newlines and tabs

NOTE: *Remove* indicates that the original text was removed (i.e. not replaced by a space, etc.)

## Alternative approaches

One of the suspected sources of error in the original work was missing signals/information from the document.
The pre-processing employed is a possible source of this missing information.
Several techniques will be employed to attempt to rectify this, including

* Extract a metric for the proportion of the document that is written in uppercase
  * This is intended to identify documents that are written in *all caps*
* Extract a metric for the number of emojis used within the document
  * This could signify exaggeration?
* Extract the base URL from hyperlinks
  * This is intended to identify sites which may be commonly attributed to malicious comments
  * NOTE: Base URL is defined as `www.example_site.ca/example/extension` -> `example_site.ca/`
* Extract a metric for the proportion of the document that uses punctuation
  * This is meant to identify documents that contain a statistically significant number of punctuation (ex. !, ?, etc.)
  * It is hypothesized that the *over-use* of punctuation (or certain punctuation) could identify aggressive behavior
* Extract words from hashtags where possible
  * Often times when a user uses a hashtag they will use CamelCase within it, making it possible to extract the words
* Conditionally replace punctuation instead of removing it
  * In certain cases punctuation (or digits) are used to take the place of letters
  * Instead of removing the symbol it could be replaced with its intended letter (ex. m@chine -> machine)

As more approaches are considered they will/should be added above.

### Implementation

The implementation of the previously identified approaches will be implemented as follows

* A simple regex can be used to find uppercase characters, sum, then replace them
* Base URLs can be extracted effectively using regex
* Metrics for several common/key symbols (i.e. !, ?, $, &, etc.)
* Again, regex can be used to identify symbols within words and replace for common ones (i.e. @, 1, 0, etc.)

**NOTE:** Before any of the above implementations are used, their effect will be considered on a sub-sample of the documents.

