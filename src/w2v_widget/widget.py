import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display
from ipywidgets import Layout


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


class WVWidget:
    def __init__(
        self, wv_model, TSNE_embedding, tokens_with_ws, initial_search_words=[]
    ):
        # Store the w2v model
        self.wv_model = wv_model
        self.vocab = wv_model.wv.index_to_key
        self.key_to_index = wv_model.wv.key_to_index
        self.TSNE_embedding = TSNE_embedding

        # Store data
        self.tokens_with_ws = tokens_with_ws

        # Store the search words
        if type(initial_search_words) == str:
            self.search_words = [initial_search_words]
            self.topic_words = [initial_search_words]
        else:
            self.search_words = initial_search_words
            self.topic_words = initial_search_words.copy()

        self.queries = {}
        self.topics = {}

    #############
    ### PLOTS ###
    #############

    def generate_plot_figure(self, title):
        """Function for creating FigureWidgets with specified title"""

        figure_widget = go.FigureWidget()

        x_range, y_range = self.wv_get_axis_range()

        figure_widget.update_xaxes(range=[min(x_range), max(x_range)])

        figure_widget.update_yaxes(range=[min(y_range), max(y_range)])

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
        x_range, y_range = (self.TSNE_embedding[:, 0], self.TSNE_embedding[:, 1])
        return x_range, y_range

    def add_word_embedding_traces(self):

        self.wv_figure_widget.data = ()

        search_keys, search_values = self.get_word_embedding_key_values(
            _filter=[
                box.description.split()[0]
                for box in self.checkboxes
                if box.disabled == False
            ]
        )

        self.wv_figure_widget.add_trace(
            go.Scatter(
                x=self.TSNE_embedding[search_values, 0],
                y=self.TSNE_embedding[search_values, 1],
                name="Similar",
                text=search_keys,
                mode="markers",
                marker=dict(color="green"),
            )
        )

        query_keys, query_values = self.get_word_embedding_key_values(
            _filter=self.search_words
        )

        self.wv_figure_widget.add_trace(
            go.Scatter(
                x=self.TSNE_embedding[query_values, 0],
                y=self.TSNE_embedding[query_values, 1],
                name="Query",
                text=query_keys,
                mode="markers",
                marker=dict(color="blue"),
            )
        )

        topic_keys, topic_values = self.get_word_embedding_key_values(
            _filter=[word for word in self.topic_words if word not in self.search_words]
        )

        self.wv_figure_widget.add_trace(
            go.Scatter(
                x=self.TSNE_embedding[topic_values, 0],
                y=self.TSNE_embedding[topic_values, 1],
                name="Topic",
                text=topic_keys,
                mode="markers",
                marker=dict(color="orange"),
            )
        )

    ### d2v ###

    def get_doc_embedding_key_values(self, _filter):
        selected_key_to_index = {
            k: v for k, v in self.key_to_index.items() if k in _filter
        }

        keys = list(selected_key_to_index.keys())
        values = list(selected_key_to_index.values())

        return keys, values

    def dv_get_axis_range(self):
        x_range, y_range = (
            self.wv_model.dv.TSNE_embedding[:, 0],
            self.wv_model.dv.TSNE_embedding[:, 1],
        )
        return x_range, y_range

    def add_document_embedding_traces(self):

        self.dv_figure_widget.data = ()

        topic_words = self.topic_words

        x, y = self.wv_model.dv.get_TSNE_reduced_doc(topic_words)

        self.dv_figure_widget.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name="Topic",
                text=topic_words,
                mode="markers",
                marker=dict(color="orange"),
            )
        )

        query_keys, query_values = self.wv_model.dv.most_similar(topic_words)

        self.dv_figure_widget.add_trace(
            go.Scatter(
                x=self.TSNE_embedding[query_values, 0],
                y=self.TSNE_embedding[query_values, 1],
                name="Documents",
                text=query_keys,
                mode="markers",
                marker=dict(color="blue"),
            )
        )

    ################
    ### ELEMENTS ###
    ################

    def generate_checkboxes(self):
        if len(self.search_words) > 0:
            return [
                widgets.Checkbox(
                    value=False,
                    description=f"{key[0]} ({round(key[1], 2)})",
                    indent=False,
                )
                for key in self.wv_model.filtered_similar(
                    self.search_words, self.topic_words
                )
            ]
        else:
            return [
                widgets.Checkbox(
                    value=False, description="", disabled=True, indent=False
                )
                for key in range(10)
            ]

    def update_checkboxes(self):
        new_words = self.wv_model.filtered_similar(self.search_words, self.topic_words)

        for i in range(10):
            self.checkboxes[i].value = False
            if i < len(new_words):
                key = new_words[i]
                self.checkboxes[i].description = f"{key[0]} ({round(key[1], 2)})"
                self.checkboxes[i].disabled = False
            else:
                self.checkboxes[i].description = ""
                self.checkboxes[i].disabled = True

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

    def create_widgets(self):

        # Add word
        self.new_search = widgets.Combobox(
            placeholder="Type a word",
            options=self.vocab,
            ensure_option=True,
            disabled=False,
        )
        self.new_search.on_submit(self.on_new_search_submit)

        self.checkboxes = self.generate_checkboxes()
        self.search_menu = self.generate_search_menu()
        self.topic_menu = self.generate_topic_menu()

        self.wv_figure_widget = self.generate_plot_figure(
            "TSNE-embedding of word2vec-space"
        )
        self.add_word_embedding_traces()
        self.dv_figure_widget = self.generate_plot_figure(
            "TSNE-embedding of doc2vec-space"
        )
        # self.add_document_embedding_traces()

        self.load_button = widgets.Button(
            description="Add",
            tooltip="Add the selected search terms to query and topic terms",
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
            description="Select", tooltip="Use the selected words as query"
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

        # Save subsample query
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

    #################
    ### CALLBACKS ###
    #################

    def on_new_search_submit(self, change):
        self.search_words = [change.value]
        self.topic_words = [change.value]

        self.new_search.value = ""
        self.update_output()

    def on_load_button_clicked(self, change):
        """
        Load functionality
        """
        new_words = [
            box.description.split()[0] for box in self.checkboxes if box.value == True
        ]

        if new_words:
            # Update the search words based on the selected boxes
            self.search_words += new_words

            self.topic_words += new_words

            self.update_output()

    def on_text_input_submit(self, change):
        self.search_words.append(change.value)
        self.topic_words.append(change.value)

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
        self.topics[topic] = list(self.search_words)

        self.save_topic.value = ""

        if topic not in self.toggle_buttons.options:
            self.toggle_buttons.options += tuple([topic])
            self.toggle_buttons.tooltips += tuple([topic])

        self.toggle_buttons.value = topic

    def on_topic_buttons_clicked(self, change):
        topic_words = list(self.topics[change.value])

        self.search_words = topic_words
        self.topic_words = topic_words.copy()

        self.update_output()

    def on_generate_sample_button_click(self, change):
        self.document_output.value = ""
        if self.document_sample_buttons.value == "Query":
            documents = [
                (i, self.html_format_text(document, self.search_words))
                for i, document in enumerate(self.tokens_with_ws)
                if len([token for token in document if token in self.search_words])
                > self.threshold.value
            ]
            self.document_samples = iter(documents)
        elif self.document_sample_buttons.value == "Topic":
            documents = [
                (i, self.html_format_text(document, self.topic_words))
                for i, document in enumerate(self.tokens_with_ws)
                if len([token for token in document if token in self.topic_words])
                > self.threshold.value
            ]
            self.document_samples = iter(documents)
        self.n_samples.value = f"Number of documents in sample: {len(documents)}"

    def on_document_sample_button_click(self, change):
        try:
            sample = next(self.document_samples)
            self.document_output.value = f"Index: {sample[0]}</br>{sample[1]}"
        except StopIteration:
            self.document_output.value = (
                "<b>No more documents with the specified threshold and query</b>"
            )

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
    margin-right: 100px;
    max-width: 200px
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

    def html_format_text(self, text, word_list):
        doc = self.nlp(text.replace("-", " "))
        return "".join(
            [
                token
                if len(token) == 0 or token.lower() not in word_list
                else f'<span style="color:teal">{token}</span>'
                for token in doc
            ]
        )

    ###############
    ### DISPALY ###
    ###############

    def display_widgets(self):
        """
        Initialize the widget with the provided search terms
        """
        # Create widget elements
        self.create_widgets()

        display(
            widgets.VBox(
                [
                    self.css,
                    widgets.HTML(
                        """<h1>w2v sampler</h1>
<p>This interface helps you build a topics from a word2vec model.</p>
<p>"""
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
                            widgets.VBox(
                                [
                                    widgets.HTML("<b>Similar words</b>"),
                                    *self.checkboxes,
                                ],
                                layout=Layout(overflow_x="hidden"),
                            ),
                            widgets.VBox(
                                [widgets.HTML("<b>Query words</b>"), self.search_menu],
                                layout=Layout(overflow_x="hidden"),
                            ),
                            widgets.VBox(
                                [widgets.HTML("<b>Topic words</b>"), self.topic_menu],
                                layout=Layout(overflow_x="hidden"),
                            ),
                            widgets.Tab(
                                [self.wv_figure_widget, self.dv_figure_widget],
                                _titles={
                                    k: v for k, v in enumerate(["Words", "Documents"])
                                },
                                layout=Layout(margin="0px 50px 0px 0px"),
                            ),
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
                            grid_template_columns="20% 20% 20% 40%",
                            grid_template_areas="""
                            "a b c d"
                            "e f g h "
                            """,
                            grid_gap="20px 10px",
                            overflow_x="hidden",
                        ),
                    ),
                    widgets.VBox(
                        [
                            widgets.HTML("""<h2>Document sample</h2>"""),
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
                ]
            )
        )
