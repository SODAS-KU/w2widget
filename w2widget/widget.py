import json
import math
import re
import tempfile
import textwrap
import time
from functools import partial
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Tuple, Union, TypedDict

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from IPython.display import Javascript, display
from ipywidgets import Layout

TEMP_FOLDER = Path(tempfile.gettempdir()) / "w2widget"

Path(TEMP_FOLDER).mkdir(exist_ok=True)


class Topic(TypedDict):
    topic_words: list[str]
    search_words: list[str]
    negative_words: list[str]
    skip_words: list[str]


class ClickResponsiveToggleButtons(widgets.ToggleButtons):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._click_handlers = widgets.CallbackDispatcher()
        self.on_msg(self._handle_button_msg)
        pass

    def on_click(self, callback, remove=False):
        """Register a callback to execute when the button is clicked.

        The callback will be called with one argument, the clicked button
        widget instance.

        Parameters
        ----------
        remove: bool (optional)
            Set to true to remove the callback from the list of callbacks.
        """
        self._click_handlers.register_callback(callback, remove=remove)

    def _handle_button_msg(self, _, content, buffers):
        """Handle a msg from the front-end.

        Parameters
        ----------
        content: dict
            Content of the msg.
        """
        if content.get("event", "") == "click":
            self._click_handlers(self)


class Widget:
    def __init__(
        self,
        wv_model,
        two_dim_word_embedding,
        tokens_with_ws: List[List[str]],
        dv_model=None,
        two_dim_doc_embedding=None,
        initial_search_words=[],
        custom_description="",
        custom_footer="",
        save_file_path: Union[str, Path] = None,
    ):
        # Store the w2v model
        self.wv_model = wv_model
        ## Add a function for fetching similar words filtered from a list of words
        self.wv_model.filtered_similar = lambda word, negative, _filter: [
            x
            for x in self.wv_model.most_similar(
                positive=word, negative=negative, topn=1000
            )
            if x[0] not in _filter
        ][:10]

        self.vocab = wv_model.index_to_key
        self.key_to_index = wv_model.key_to_index

        self.two_dim_word_embedding = two_dim_word_embedding

        # store the d2v model
        self.dv_model = dv_model
        self.two_dim_doc_embedding = two_dim_doc_embedding

        # Store data
        self.tokens_with_ws = tokens_with_ws

        # Store the search words
        if type(initial_search_words) == str:
            self.search_words = [initial_search_words]
            self.topic_words = [initial_search_words]
        else:
            self.search_words = initial_search_words
            self.topic_words = initial_search_words.copy()

        self.custom_description = custom_description
        self.custom_footer = custom_footer

        if not save_file_path:
            self.save_file_path = Path(tempfile.mkstemp(dir=str(TEMP_FOLDER))[1])
        else:
            self.save_file_path = Path(save_file_path).absolute

        self.negative_words = []
        self.skip_words = []
        self.queries = {}
        self.topics = {}

        # Create widget elements
        self.create_widgets()

    def load_topics(self, topics: Dict[str, Topic]):
        self.topics = topics

        for topic in topics:
            if topic not in self.toggle_buttons.options:
                self.toggle_buttons.options += tuple([topic])
                self.toggle_buttons.tooltips += tuple([topic])

    #############
    ### PLOTS ###
    #############

    def generate_plot_figure(
        self, title: str, xy_range: Tuple[Tuple[int, int], Tuple[int, int]]
    ):
        """Function for creating FigureWidgets with specified title"""

        figure_widget = go.FigureWidget()

        x_range, y_range = xy_range

        figure_widget.update_xaxes(range=[min(x_range) * 1.05, max(x_range) * 1.05])

        figure_widget.update_yaxes(range=[min(y_range) * 1.05, max(y_range) * 1.05])

        figure_widget.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            title=title,
            legend=dict(
                font=dict(
                    size=15,
                ),
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
            ),
        )

        return figure_widget

    ### w2v ###

    def get_word_embedding_key_values(self, _filter):
        selected_key_to_index = {
            k: v for k, v in self.key_to_index.items() if k in _filter
        }

        keys = list(selected_key_to_index.keys())
        values = list(selected_key_to_index.values())

        return keys, values

    def wv_get_axis_range(self):
        x_range, y_range = (
            self.two_dim_word_embedding[:, 0],
            self.two_dim_word_embedding[:, 1],
        )
        return x_range, y_range

    def add_word_embedding_traces(self):

        self.wv_figure_widget.data = ()

        # For similar but not selected
        search_keys, search_values = self.get_word_embedding_key_values(
            _filter=self.get_text_labels()
        )

        self.wv_figure_widget.add_trace(
            go.Scatter(
                x=self.two_dim_word_embedding[search_values, 0],
                y=self.two_dim_word_embedding[search_values, 1],
                name="Similar",
                text=search_keys,
                mode="markers",
                marker=dict(color="green"),
            )
        )

        # For current query
        query_keys, query_values = self.get_word_embedding_key_values(
            _filter=self.search_words
        )

        self.wv_figure_widget.add_trace(
            go.Scatter(
                x=self.two_dim_word_embedding[query_values, 0],
                y=self.two_dim_word_embedding[query_values, 1],
                name="Query",
                text=query_keys,
                mode="markers",
                marker=dict(color="blue"),
            )
        )

        # For current topic
        topic_keys, topic_values = self.get_word_embedding_key_values(
            _filter=[word for word in self.topic_words if word not in self.search_words]
        )

        self.wv_figure_widget.add_trace(
            go.Scatter(
                x=self.two_dim_word_embedding[topic_values, 0],
                y=self.two_dim_word_embedding[topic_values, 1],
                name="Topic",
                text=topic_keys,
                mode="markers",
                marker=dict(color="orange"),
            )
        )

        # For skipped
        topic_keys, topic_values = self.get_word_embedding_key_values(
            _filter=self.skip_words
        )

        self.wv_figure_widget.add_trace(
            go.Scatter(
                x=self.two_dim_word_embedding[topic_values, 0],
                y=self.two_dim_word_embedding[topic_values, 1],
                name="Skipped",
                text=topic_keys,
                mode="markers",
                marker=dict(color="grey"),
            )
        )

        # For negative
        topic_keys, topic_values = self.get_word_embedding_key_values(
            _filter=self.negative_words
        )

        self.wv_figure_widget.add_trace(
            go.Scatter(
                x=self.two_dim_word_embedding[topic_values, 0],
                y=self.two_dim_word_embedding[topic_values, 1],
                name="Not",
                text=topic_keys,
                mode="markers",
                marker=dict(color="red"),
            )
        )

    ### d2v ###

    def dv_get_axis_range(self):
        x_range, y_range = (
            self.two_dim_doc_embedding[:, 0],
            self.two_dim_doc_embedding[:, 1],
        )
        return x_range, y_range

    def add_document_embedding_traces(self):

        self.dv_figure_widget.data = ()

        topic_words = self.topic_words.copy()

        if topic_words:
            x, y = self.dv_model.get_TSNE_reduced_doc(topic_words)[0]

            self.dv_figure_widget.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    name="Topic",
                    text=str(topic_words),
                    mode="markers",
                    marker=dict(color="orange"),
                )
            )

            query_values = [x[0] for x in self.dv_model.most_similar(topic_words)]
            query_keys = np.array(self.tokens_with_ws, dtype="object")[query_values]

            self.dv_figure_widget.add_trace(
                go.Scatter(
                    x=self.two_dim_doc_embedding[query_values, 0],
                    y=self.two_dim_doc_embedding[query_values, 1],
                    name="Documents",
                    text=[
                        f"Document index: {i}</br></br>{self.html_format_text(text, topic_words, text_wrap=True)}"
                        for i, text in zip(query_values, query_keys.tolist())
                    ],
                    hoverinfo="text",
                    hoverlabel=dict(namelength=-1),
                    mode="markers",
                    marker=dict(color="blue"),
                )
            )

    ################
    ### ELEMENTS ###
    ################

    def generate_checkboxes_text(self):
        if len(self.search_words) > 0:
            return [
                widgets.Label(
                    value=f"{key[0]} ({round(key[1], 2)})",
                )
                for key in self.wv_model.filtered_similar(
                    self.search_words,
                    self.negative_words,
                    self.topic_words + self.skip_words,
                )
            ]
        else:
            return [
                widgets.Label(
                    value="",
                )
                for key in range(10)
            ]

    def generate_accept_checkboxes(self):
        if len(self.search_words) > 0:
            return [
                widgets.Checkbox(
                    value=False,
                    description="",
                    indent=False,
                )
                for key in self.wv_model.filtered_similar(
                    self.search_words,
                    self.negative_words,
                    self.topic_words + self.skip_words,
                )
            ]
        else:
            return [
                widgets.Checkbox(
                    value=False, description="", disabled=True, indent=False
                )
                for key in range(10)
            ]

    def generate_not_checkboxes(self):
        if len(self.search_words) > 0:
            return [
                widgets.Checkbox(
                    value=False,
                    description="",
                    indent=False,
                )
                for key in self.wv_model.filtered_similar(
                    self.search_words,
                    self.negative_words,
                    self.topic_words + self.skip_words,
                )
            ]
        else:
            return [
                widgets.Checkbox(
                    value=False, description="", disabled=True, indent=False
                )
                for key in range(10)
            ]

    def generate_skip_checkboxes(self):
        if len(self.search_words) > 0:
            return [
                widgets.Checkbox(
                    value=False,
                    description="",
                    indent=False,
                )
                for key in self.wv_model.filtered_similar(
                    self.search_words,
                    self.negative_words,
                    self.topic_words + self.skip_words,
                )
            ]
        else:
            return [
                widgets.Checkbox(
                    value=False, description="", disabled=True, indent=False
                )
                for key in range(10)
            ]

    def get_text_labels(self):
        return [label.value.split()[0] for label in self.checkboxes_text if label.value]

    def get_accepted(self):
        return [
            x.value.split()[0]
            for n, x in enumerate(self.checkboxes_text)
            if self.accept_checkboxes[n].value
        ]

    def get_not(self):
        return [
            x.value.split()[0]
            for n, x in enumerate(self.checkboxes_text)
            if self.not_checkboxes[n].value
        ]

    def get_skipped(self):
        return [
            x.value.split()[0]
            for n, x in enumerate(self.checkboxes_text)
            if self.skip_checkboxes[n].value
        ]

    def update_checkboxes(self):
        new_words = self.wv_model.filtered_similar(
            self.search_words, self.negative_words, self.topic_words + self.skip_words
        )

        for i in range(10):
            self.accept_checkboxes[i].value = False
            self.not_checkboxes[i].value = False
            self.skip_checkboxes[i].value = False
            if i < len(new_words):
                key = new_words[i]
                self.checkboxes_text[i].value = f"{key[0]} ({round(key[1], 2)})"

                self.accept_checkboxes[i].disabled = False
                self.not_checkboxes[i].disabled = False
                self.skip_checkboxes[i].disabled = False
            else:
                self.checkboxes_text[i].value = ""

                self.accept_checkboxes[i].disabled = True
                self.not_checkboxes[i].disabled = True
                self.skip_checkboxes[i].disabled = True

    def generate_search_menu(self):
        return widgets.SelectMultiple(
            options=[word for word in self.search_words],
            value=[],
            rows=18,
            disabled=False,
        )

    def update_search_menu(self):
        self.search_menu.options = self.search_words

    def generate_topic_menu(self):
        return widgets.SelectMultiple(
            options=[word for word in self.topic_words],
            value=[],
            rows=18,
            disabled=False,
        )

    def update_topic_menu(self):
        self.topic_menu.options = self.topic_words

    def add_checkbox_observe(self, n, checkboxes):
        n_classes = len(checkboxes[0])
        class_n = math.ceil((n + 1) / n_classes) - 1
        children_n = n % n_classes
        checkbox = [c for C in checkboxes for c in C][n]

        checkboxes = checkboxes.copy()
        checkboxes.pop(class_n)

        observe_function = partial(
            self.on_checkbox,
            checkboxes=[checkbox[children_n] for checkbox in checkboxes],
        )

        checkbox.observe(observe_function)

    def on_checkbox(self, c, checkboxes):
        for checkbox in checkboxes:
            checkbox.value = False

    def create_widgets(self):

        # Add word
        self.new_search = widgets.Combobox(
            placeholder="Type a word",
            options=self.vocab,
            ensure_option=True,
            disabled=False,
        )
        self.new_search.on_submit(self.on_new_search_submit)

        self.checkboxes_text = self.generate_checkboxes_text()

        self.accept_checkboxes = self.generate_accept_checkboxes()
        self.not_checkboxes = self.generate_not_checkboxes()
        self.skip_checkboxes = self.generate_skip_checkboxes()

        checkboxes = [
            self.accept_checkboxes,
            self.not_checkboxes,
            self.skip_checkboxes,
        ]

        for n in range(len([c for C in checkboxes for c in C])):
            self.add_checkbox_observe(n, checkboxes)

        self.search_menu = self.generate_search_menu()
        self.topic_menu = self.generate_topic_menu()

        self.wv_figure_widget = self.generate_plot_figure(
            "Embedding of word2vec-space", xy_range=self.wv_get_axis_range()
        )
        self.add_word_embedding_traces()

        if self.dv_model:
            self.dv_figure_widget = self.generate_plot_figure(
                "Embedding of doc2vec-space", xy_range=self.dv_get_axis_range()
            )
            self.add_document_embedding_traces()

            self.plot_tab = widgets.Tab(
                children=[self.wv_figure_widget, self.dv_figure_widget],
                layout=Layout(margin="0px 50px 0px 0px"),
            )
            self.plot_tab._titles = {0: "Words", 1: "Documents"}
            self.plot_tab.observe(self.on_tabs_change, names="selected_index")
        else:
            self.plot_tab = widgets.Box((self.wv_figure_widget,))
            self.plot_tab.layout = Layout(margin="0px 50px 0px 0px")

        self.load_button = widgets.Button(
            description="Next",
            tooltip="Add the selected search terms to query and topic terms and get the next sample of similar words",
        )
        self.load_button.on_click(self.on_load_button_clicked)

        # Add word
        self.text_input = widgets.Combobox(
            placeholder="Add a word from the model",
            options=self.vocab,
            ensure_option=True,
            disabled=False,
        )
        self.text_input.on_submit(self.on_text_input_submit)

        self.query_select_button = widgets.Button(
            description="Filter", tooltip="Use the selected words as query"
        )
        self.query_select_button.on_click(self.on_query_select_button_clicked)

        self.query_remove_button = widgets.Button(
            description="Remove", tooltip="Remove the selected words from query"
        )
        self.query_remove_button.on_click(self.on_query_remove_button_clicked)

        # Save subsample query
        self.save_query = widgets.Text(placeholder="Name of query", disabled=False)
        self.save_query.on_submit(self.on_save_query_submit)

        self.topic_query_button = widgets.Button(
            description="Query", tooltip="Use the selected words as query"
        )
        self.topic_query_button.on_click(self.on_topic_query_button_clicked)

        self.topic_remove_button = widgets.Button(
            description="Remove",
            tooltip="Remove the selected words from the topic and query",
        )
        self.topic_remove_button.on_click(self.on_topic_remove_button_clicked)

        # Save subsample topic
        self.save_topic = widgets.Text(placeholder="Name of topic", disabled=False)
        self.save_topic.on_submit(self.on_save_topic_submit)

        self.toggle_buttons = ClickResponsiveToggleButtons(
            options=[],
            button_style="info",
        )
        self.toggle_buttons.on_click(self.on_topic_buttons_clicked)

        self.threshold = widgets.IntText(
            value=5,
            description="Threshold:",
            disabled=False,
            indent=False,
        )

        self.document_sample_buttons = ClickResponsiveToggleButtons(
            options=["Query", "Topic"],
            value=None,
        )

        self.generate_sample_button = widgets.Button(description="Generate samples")
        self.generate_sample_button.on_click(self.on_generate_sample_button_click)

        self.n_samples = widgets.HTML()

        self.document_sample_button = widgets.Button(description="Sample")
        self.document_sample_button.on_click(self.on_document_sample_button_click)

        self.filename = widgets.Text(
            placeholder="Filename",
            value=""
            if self.save_file_path.parent == TEMP_FOLDER
            else self.save_file_path.name,
            disabled=False,
        )
        self.filename.on_submit(self.on_filename_submit)

        self.save_file = widgets.Button(
            description="Save file",
            tooltip="Save the topics as a json file with the specified file name",
        )
        self.save_file.on_click(self.on_save_file_click)

        self.document_output = widgets.HTML()

        self.css = self.generate_css()

    def update_output(self):
        # Create new checkboxes based on the new most similar words
        self.update_checkboxes()

        # Update search menu
        self.update_search_menu()

        # Update topic menu
        self.update_topic_menu()

        # Word embedding plot
        self.add_word_embedding_traces()
        self.add_document_embedding_traces()

    #################
    ### CALLBACKS ###
    #################

    def on_new_search_submit(self, change):
        self.search_words = [change.value]
        self.topic_words = [change.value]

        self.new_search.value = ""
        self.negative_words = []
        self.skip_words = []

        self.update_output()
        self.save_data()

    def on_load_button_clicked(self, change):
        """
        Load functionality
        """
        change = False
        accepted_words = self.get_accepted()
        not_words = self.get_not()
        skip_words = self.get_skipped()

        if accepted_words:
            # Update the search words based on the selected boxes
            self.search_words += accepted_words

            self.topic_words += accepted_words
            change = True

        if not_words:
            self.negative_words += not_words
            change = True

        if skip_words:
            self.skip_words += skip_words
            change = True

        if change:
            self.update_output()

    def on_text_input_submit(self, change):
        word = change.value
        if word:
            if word not in self.topic_words:
                self.topic_words.append(change.value)
            if word not in self.search_words:
                self.search_words.append(change.value)

            self.text_input.value = ""
            self.update_output()

    def on_save_query_submit(self, change):
        self.queries[change.value] = list(self.search_words)
        self.save_query.value = ""

    def on_query_remove_button_clicked(self, change):
        self.search_words = [
            word for word in self.search_words if word not in self.search_menu.value
        ]

        self.update_output()

    def on_query_select_button_clicked(self, change):
        """ """
        self.search_words = list(self.search_menu.value)

        self.update_output()

    def on_topic_query_button_clicked(self, change):
        self.search_words = list(self.topic_menu.value)

        self.update_output()

    def on_topic_remove_button_clicked(self, change):
        self.topic_words = [
            word for word in self.topic_words if word not in self.topic_menu.value
        ]
        self.search_words = [
            word for word in self.search_words if word not in self.topic_menu.value
        ]

        self.update_output()

    def on_save_topic_submit(self, change):
        topic = change.value.title()

        if topic:
            self.topics[topic] = {
                "topic_words": list(self.topic_words),
                "search_words": list(self.search_words),
                "negative_words": list(self.negative_words),
                "skip_words": list(self.skip_words),
            }

            self.save_topic.value = ""

            if topic not in self.toggle_buttons.options:
                self.toggle_buttons.options += tuple([topic])
                self.toggle_buttons.tooltips += tuple([topic])

            self.toggle_buttons.value = topic

    def on_topic_buttons_clicked(self, change):
        topic = self.topics[change.value]

        self.search_words = list(topic["search_words"])
        self.topic_words = list(topic["topic_words"])

        self.negative_words = list(topic["negative_words"])
        self.skip_words = list(topic["skip_words"])

        self.update_output()

    def on_tabs_change(self, change):
        self.dv_figure_widget.layout.autosize = True
        self.update_output()

    def on_generate_sample_button_click(self, change):
        self.document_output.value = ""

        if self.document_sample_buttons.value == "Query":
            documents = [
                (i, self.html_format_text(document, self.search_words))
                for i, document in enumerate(self.tokens_with_ws)
                if len(
                    [token for token in document if token.lower() in self.search_words]
                )
                >= self.threshold.value
            ]
            self.document_samples = cycle(documents)
        elif self.document_sample_buttons.value == "Topic":
            documents = [
                (i, self.html_format_text(document, self.topic_words))
                for i, document in enumerate(self.tokens_with_ws)
                if len(
                    [token for token in document if token.lower() in self.topic_words]
                )
                >= self.threshold.value
            ]
            self.document_samples = cycle(documents)

        self.n_samples.value = f"Number of documents in sample: {len(documents)}"

    def on_document_sample_button_click(self, change):
        try:
            sample = next(self.document_samples)
            self.document_output.value = (
                f"Document index: {sample[0]}</br></br>{sample[1]}"
            )
        except StopIteration:
            self.document_output.value = (
                "<b>No more documents with the specified threshold and query</b>"
            )

    def on_filename_submit(self, change):
        filename = change.value

        if len(filename) < 1:
            return
        else:
            with open(change.value + ".json", "w") as f:
                json.dump(self.topics, f)
        time.sleep(0.3)
        self.filename.value = ""

    def on_save_file_click(self, change):
        filename = self.filename.value
        if len(filename) < 1:
            return
        else:
            self.save_data(filename)
        time.sleep(0.3)
        self.filename.value = ""

    def save_data(self, filename=None):
        if filename:
            with open(filename + ".json", "w") as f:
                json.dump(self.topics, f)
        else:
            with open(self.save_file_path + ".json", "w") as f:
                json.dump(self.topics, f)

    ###########
    ### CSS ###
    ###########

    def generate_css(self):
        return widgets.HTML(
            """<style>
.widget-button {
    margin-right: 160px;
}

.widget-select-multiple {
    max-width: 200px;
    width:auto;
}

.widget-output {
    min-height: 475px;
}

option {
    padding-left: 5px;
}

.widget-text {
    width:190px
}

.widget-toggle-buttons {
    max-width:700px
}

</style>"""
        )

    #############
    ### UTILS ###
    #############

    def html_format_text(
        self, tokens_with_ws: List[str], word_list: List[str], text_wrap=False
    ):

        if text_wrap:
            text = " ".join(
                [
                    token
                    if len(token) == 0
                    or re.sub(r"[^\w]", "", token).lower() not in word_list
                    else f'<span style="color:orange">{token}</span>'
                    for token in " </br> ".join(
                        textwrap.wrap("".join(tokens_with_ws), max_lines=7)
                    ).split()
                ]
            ).replace(" </br> ", "</br>")

        else:
            text = "".join(
                [
                    token
                    if len(token) == 0 or token.lower() not in word_list
                    else f'<span style="background:yellow;opacity:0.8;z-index:-1">{token}</span>'
                    for token in tokens_with_ws
                ]
            )
        return text

    ###############
    ### DISPALY ###
    ###############

    def display_widget(self):
        """
        Initialize the widget with the provided search terms
        """

        self.wv_figure_widget.layout.autosize = True

        self.view = widgets.VBox(
            [
                self.css,
                widgets.HTML(
                    f"""<h1>w2widget</h1>
<p>Inspired by the <a href="https://journals.sagepub.com/doi/full/10.1177/20539517221080146">CALM framework</a>, this interface helps you build topics from a word2vec model.
</br>
This exercise may help the researcher in gaining insight in a large text corpus, and works best with a concurrent qualitative reading of the documents.
</p>
<p>The workflow in this widget is build around 5 different elements:</p>
<ul>
    <li><u>Word similarities</u> calculated from queries.</li>
    <li><u>Queries</u> consisting of <i>positive</i> and <i>negative</i> words.</li>
    <li><u>Topics</u> consisting of all the words you have accepted.</li>
    <li><u>Plot</u> visualizing the position of the word vectors in a two-dimensional space.</li>
    <li><u>Document sampling</u> sample documents from query or topic to get an understanding of the semantic context.</li>
</ul>
{self.custom_description}
<hr>
<h2>Topic generation</h2>"""
                ),
                widgets.VBox(
                    [
                        widgets.HTML("Start a new query from the specified word"),
                        self.new_search,
                        widgets.HTML("</br>"),
                    ]
                ),
                widgets.GridBox(
                    children=[
                        widgets.GridBox(
                            [
                                widgets.VBox(
                                    [
                                        widgets.HTML("<b>Similar words</b>"),
                                        *self.checkboxes_text,
                                    ],
                                    layout=Layout(overflow_x="hidden"),
                                ),
                                widgets.VBox(
                                    [
                                        widgets.HTML("<b>Accept</b>"),
                                        *self.accept_checkboxes,
                                    ],
                                    layout=Layout(overflow_x="hidden"),
                                ),
                                widgets.VBox(
                                    [widgets.HTML("<b>Not</b>"), *self.not_checkboxes],
                                    layout=Layout(overflow_x="hidden"),
                                ),
                                widgets.VBox(
                                    [
                                        widgets.HTML("<b>Skip</b>"),
                                        *self.skip_checkboxes,
                                    ],
                                    layout=Layout(overflow_x="hidden"),
                                ),
                            ],
                            layout=Layout(
                                width="100%",
                                grid_template_rows="auto",
                                grid_template_columns="50% 15% 15% 20%",
                                grid_template_areas="""
                                    "a b c d"
                                    """,
                                grid_gap="20px 10px",
                                overflow_x="hidden",
                                padding="0 10% 0 0",
                            ),
                        ),
                        widgets.VBox(
                            [widgets.HTML("<b>Query words</b>"), self.search_menu],
                            layout=Layout(overflow_x="hidden"),
                        ),
                        widgets.VBox(
                            [widgets.HTML("<b>Topic words</b>"), self.topic_menu],
                            layout=Layout(overflow_x="hidden"),
                        ),
                        self.plot_tab,
                        widgets.VBox(
                            [
                                self.load_button,
                                widgets.HTML("Add a word:"),
                                self.text_input,
                            ],
                            layout=Layout(overflow_x="hidden"),
                        ),
                        widgets.VBox(
                            [
                                self.query_select_button,
                                self.query_remove_button,
                                widgets.HTML("Save the current query:"),
                                self.save_query,
                            ],
                            layout=Layout(overflow_x="hidden"),
                        ),
                        widgets.VBox(
                            [
                                self.topic_query_button,
                                self.topic_remove_button,
                                widgets.HTML("Save the current topic:"),
                                self.save_topic,
                            ],
                            layout=Layout(overflow_x="hidden"),
                        ),
                        widgets.VBox(
                            [
                                widgets.HTML(
                                    "<h2>Topics</h2>",
                                    layout=Layout(margin="-20px 0px 0px 0px"),
                                ),
                                self.toggle_buttons,
                            ],
                            layout=Layout(overflow_x="hidden"),
                        ),
                    ],
                    layout=Layout(
                        width="100%",
                        grid_template_rows="auto auto auto",
                        grid_template_columns="30% 15% 15% 40%",
                        grid_template_areas="""
                            "a b c d"
                            "e f g h "
                            """,
                        grid_gap="20px 10px",
                        overflow_x="hidden",
                    ),
                ),
                widgets.HTML("<hr>"),
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                widgets.HTML("""<h2>Document sampling</h2>"""),
                                widgets.HTML(
                                    "Sample documents from either query or topic word list"
                                ),
                                widgets.HBox(
                                    [
                                        self.threshold,
                                        self.document_sample_buttons,
                                        self.generate_sample_button,
                                    ]
                                ),
                                self.n_samples,
                                widgets.VBox(
                                    [
                                        self.document_sample_button,
                                        widgets.Box(
                                            [self.document_output],
                                            layout=Layout(
                                                height="200px", display="overflow"
                                            ),
                                        ),
                                    ],
                                    layout=Layout(margin="20px", width="650px"),
                                ),
                            ]
                        ),
                        widgets.VBox(
                            [
                                widgets.HTML("""<h2>Save topics</h2>"""),
                                widgets.HBox(
                                    [
                                        self.filename,
                                        self.save_file,
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
                widgets.HTML(self.custom_footer),
            ]
        )

        display(self.view)

        # Javascript injection to help plotly graph fit it's container
        display(
            Javascript(
                """
                // Javascript injection to help plotly graph fit it's container
                selector = "div.js-plotly-plot:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(4) > div:nth-child(1) > div:nth-child(3) > a:nth-child(4)";
                
                function waitForElementToDisplay(selector, callback, checkFrequencyInMs, timeoutInMs) {
                    var startTimeInMs = Date.now();
                    (function loopSearch() {
                        if (document.querySelector(selector) != null) {
                        callback();
                        return;
                        }
                        else {
                        setTimeout(function () {
                            if (timeoutInMs && Date.now() - startTimeInMs > timeoutInMs)
                            return;
                            loopSearch();
                        }, checkFrequencyInMs);
                        }
                    })();
                }
                
                waitForElementToDisplay(selector,function(){document.querySelector(selector).click()},1000,9000);
                console.warn('test')
            """
            )
        )

        self.wv_figure_widget.layout.autosize = True
