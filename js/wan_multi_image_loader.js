import { app } from "../../scripts/app.js";

const MAX_IMAGES = 50;
const RENDER_BATCH_SIZE = 6;
const INIT_DELAY = 200;
const RENDER_DELAY = 30;

function buildImageUrl(item) {
    const params = new URLSearchParams({
        filename: item.name || "",
        subfolder: item.subfolder || "",
        type: item.type || "input",
    });
    return `/view?${params.toString()}`;
}

class WanMultiImageGallery {
    constructor(node, imagesDataWidget) {
        this.node = node;
        this.imagesDataWidget = imagesDataWidget;

        /** @type {{name:string, type:string, subfolder:string}[]} */
        this.images = [];
        this.sortOrders = {};
        this.thumbnailElements = new Map();
        this.currentIndex = 0;

        this.isInitializing = false;
        this.initCancelled = false;
        this.renderScheduled = false;

        this.root = this._buildRootDOM();
        this._bindIndexWidget();
    }

    _buildRootDOM() {
        const container = document.createElement("div");
        container.style.cssText = `
            width: 100%;
            padding: 6px;
            background: #141414;
            border-radius: 6px;
            box-sizing: border-box;
        `;

        const btnRow = document.createElement("div");
        btnRow.style.cssText = `
            display: flex;
            gap: 6px;
            margin-bottom: 6px;
        `;

        this.uploadBtn = document.createElement("button");
        this.uploadBtn.textContent = "📁 选择";
        this.uploadBtn.style.cssText = this._buttonStyle("#2a2a2a", "#444");

        this.addBtn = document.createElement("button");
        this.addBtn.textContent = "➕ 增加";
        this.addBtn.style.cssText = this._buttonStyle("#244a24", "#4a6");

        this.sortBtn = document.createElement("button");
        this.sortBtn.textContent = "🔃 排序";
        this.sortBtn.style.cssText = this._buttonStyle("#222a4a", "#46a");

        this.clearBtn = document.createElement("button");
        this.clearBtn.textContent = "🗑️";
        this.clearBtn.style.cssText = this._buttonStyle("#4a2222", "#a44");

        btnRow.appendChild(this.uploadBtn);
        btnRow.appendChild(this.addBtn);
        btnRow.appendChild(this.sortBtn);
        btnRow.appendChild(this.clearBtn);

        this.progressBar = document.createElement("div");
        this.progressBar.style.cssText = `
            display: none;
            height: 3px;
            border-radius: 2px;
            background: #333;
            overflow: hidden;
            margin-bottom: 6px;
        `;
        this.progressFill = document.createElement("div");
        this.progressFill.style.cssText = `
            height: 100%;
            width: 0%;
        `;
        this.progressBar.appendChild(this.progressFill);

        this.previewContainer = document.createElement("div");
        this.previewContainer.style.cssText = `
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
            gap: 6px;
            max-height: 280px;
            overflow-y: auto;
            background: #1f1f1f;
            border-radius: 4px;
            padding: 4px;
        `;

        this.fileInput = document.createElement("input");
        this.fileInput.type = "file";
        this.fileInput.multiple = true;
        this.fileInput.accept = "image/*";
        this.fileInput.style.display = "none";

        container.appendChild(btnRow);
        container.appendChild(this.progressBar);
        container.appendChild(this.previewContainer);
        container.appendChild(this.fileInput);

        this._wireEvents();

        return container;
    }

    _buttonStyle(bg, border) {
        return `
            flex: 1;
            padding: 6px 8px;
            background: ${bg};
            border: 1px solid ${border};
            border-radius: 4px;
            color: #eee;
            font-size: 12px;
            cursor: pointer;
            white-space: nowrap;
        `;
    }

    _wireEvents() {
        let isAddingMode = false;

        this.uploadBtn.onclick = () => {
            isAddingMode = false;
            this.fileInput.click();
        };

        this.addBtn.onclick = () => {
            isAddingMode = true;
            this.fileInput.click();
        };

        this.sortBtn.onclick = () => this.applySorting();

        this.clearBtn.onclick = async () => {
            const confirmDialog = app?.extensionManager?.dialog?.confirm;
            let confirmed = false;

            if (confirmDialog) {
                confirmed = await confirmDialog({
                    title: "清空确认",
                    message: "确定要清空所有图片吗？",
                });
            } else {
                confirmed = window.confirm("确定要清空所有图片吗？");
            }

            if (!confirmed) return;
            this._clearAll();
        };

        this.fileInput.onchange = async (e) => {
            const files = Array.from(e.target.files);
            if (!files.length) return;
            await this._processFiles(files, !isAddingMode);
            this.fileInput.value = "";
            isAddingMode = false;
        };
    }

    _bindIndexWidget() {
        const indexWidget = this.node.widgets?.find((w) => w.name === "index");
        if (!indexWidget) return;

        this.currentIndex = indexWidget.value ?? 0;

        const originalCb = indexWidget.callback;
        const gallery = this;
        indexWidget.callback = function(...args) {
            if (originalCb) {
                originalCb.apply(this, args);
            }
            gallery.currentIndex = this.value ?? 0;
            gallery._updateHighlight();
        };
    }



    _parseWidgetValue() {
        const v = this.imagesDataWidget.value;
        if (!v || v === "[]") return [];

        try {
            const parsed = JSON.parse(v);
            if (Array.isArray(parsed)) {
                return parsed;
            }
            return [];
        } catch (e) {
            console.warn("WanMultiImageLoader: parse images_data failed", e);
            return [];
        }
    }

    _syncWidgetFromImages() {
        const slim = this.images.map((item) => ({
            name: item.name,
            type: item.type || "input",
            subfolder: item.subfolder || "",
        }));
        this.imagesDataWidget.value = JSON.stringify(slim);
    }

    async _initializeFromWidget() {
        if (this.isInitializing) return;
        this.isInitializing = true;
        this.initCancelled = false;

        try {
            const data = this._parseWidgetValue();
            if (!data || !data.length) {
                this.images = [];
                this._renderAllThumbnails();
                return;
            }

            this.images = data.map((d) => ({
                name: d.name,
                type: d.type || "input",
                subfolder: d.subfolder || "",
            }));

            this.previewContainer.innerHTML = "";
            this.thumbnailElements.clear();
            this._showProgress(true);

            const total = this.images.length;
            for (let i = 0; i < total && !this.initCancelled; i += RENDER_BATCH_SIZE) {
                const batch = this.images.slice(i, i + RENDER_BATCH_SIZE);
                const fragment = document.createDocumentFragment();

                batch.forEach((item, batchIdx) => {
                    const idx = i + batchIdx;
                    const url = buildImageUrl(item);
                    const dom = this._createThumbnail(idx, url);
                    fragment.appendChild(dom);
                });

                if (this.initCancelled) break;

                this.previewContainer.appendChild(fragment);
                this._setProgress((i + batch.length) / total);

                if (i + batch.length < total) {
                    await this._sleep(RENDER_DELAY);
                }
            }

            if (!this.initCancelled) {
                this._syncWidgetFromImages();
                const indexWidget = this.node.widgets?.find((w) => w.name === "index");
                if (indexWidget) {
                    this.currentIndex = indexWidget.value ?? 0;
                }
                this._updateHighlight();
            }
        } catch (e) {
            console.error("WanMultiImageLoader init error:", e);
            this.images = [];
            this._syncWidgetFromImages();
        } finally {
            this.isInitializing = false;
            if (!this.initCancelled) this._showProgress(false);
        }
    }

    async _processFiles(files, replace) {
        if (replace) {
            this.images = [];
            this.sortOrders = {};
            this.previewContainer.innerHTML = "";
            this.thumbnailElements.clear();
        }

        if (this.images.length + files.length > MAX_IMAGES) {
            const confirmDialog = app?.extensionManager?.dialog?.confirm;
            const excess = this.images.length + files.length - MAX_IMAGES;
            let confirmed = false;

            const msg = `最多只能存 ${MAX_IMAGES} 张，当前已有 ${this.images.length} 张，继续将截断多余 ${excess} 张，是否继续？`;
            if (confirmDialog) {
                confirmed = await confirmDialog({ title: "图片数量限制", message: msg });
            } else {
                confirmed = window.confirm(msg);
            }
            if (!confirmed) return;

            files = files.slice(0, files.length - excess);
        }

        if (!files.length) return;

        this._showProgress(true);
        this.uploadBtn.disabled = true;
        this.addBtn.disabled = true;

        const startIndex = this.images.length;
        const total = files.length;

        for (let i = 0; i < total; i += RENDER_BATCH_SIZE) {
            const batch = files.slice(i, i + RENDER_BATCH_SIZE);
            const fragment = document.createDocumentFragment();

            const results = await Promise.all(
                batch.map(async (file, batchIdx) => {
                    const idx = startIndex + i + batchIdx;
                    try {
                        const meta = await this._uploadFileToServer(file);
                        if (!meta) {
                            console.warn("WanMultiImageLoader: 上传失败", file);
                            return null;
                        }
                        const item = {
                            name: meta.name,
                            type: meta.type || "input",
                            subfolder: meta.subfolder || "",
                        };
                        this.images[idx] = item;
                        const url = buildImageUrl(item);
                        return { idx, url };
                    } catch (e) {
                        console.error("WanMultiImageLoader: 处理图片出错", file, e);
                        return null;
                    }
                }),
            );

            for (const r of results) {
                if (!r) continue;
                const dom = this._createThumbnail(r.idx, r.url);
                fragment.appendChild(dom);
            }

            this.previewContainer.appendChild(fragment);
            this._setProgress((i + batch.length) / total);

            if (i + batch.length < total) {
                await this._sleep(RENDER_DELAY);
            }
        }

        this._syncWidgetFromImages();
        this._updateHighlight();
        this._showProgress(false);
        this.uploadBtn.disabled = false;
        this.addBtn.disabled = false;
    }

    async _uploadFileToServer(file) {
        const formData = new FormData();
        formData.append("image", file, file.name);

        const res = await fetch("/upload/image", {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            console.error("WanMultiImageLoader: /upload/image 失败", res.status);
            return null;
        }

        const json = await res.json();
        return json;
    }

    _createThumbnail(index, imageUrl) {
        const wrapper = document.createElement("div");
        wrapper.style.cssText = `
            display: flex;
            flex-direction: column;
            gap: 3px;
        `;

        const thumb = document.createElement("div");
        thumb.style.cssText = `
            position: relative;
            aspect-ratio: 1;
            border-radius: 4px;
            overflow: hidden;
            cursor: pointer;
            border: 2px solid transparent;
            background: #000;
        `;

        const img = document.createElement("img");
        img.src = imageUrl;
        img.style.cssText = `
            width: 100%;
            height: 100%;
            object-fit: cover;
        `;
        img.loading = "lazy";

        const label = document.createElement("div");
        label.textContent = `#${index}`;
        label.style.cssText = `
            position: absolute;
            left: 2px;
            top: 2px;
            padding: 1px 3px;
            font-size: 10px;
            background: rgba(0,0,0,0.7);
            border-radius: 2px;
            color: #fff;
        `;

        const delBtn = document.createElement("button");
        delBtn.textContent = "×";
        delBtn.style.cssText = `
            position: absolute;
            right: 2px;
            top: 2px;
            width: 18px;
            height: 18px;
            border: none;
            border-radius: 3px;
            font-size: 14px;
            line-height: 1;
            background: rgba(255,0,0,0.7);
            color: #fff;
            cursor: pointer;
        `;

        delBtn.onclick = (e) => {
            e.stopPropagation();
            this._deleteImage(index);
        };

        thumb.onclick = () => {
            const indexWidget = this.node.widgets?.find((w) => w.name === "index");
            if (indexWidget) {
                indexWidget.value = index;
                this.currentIndex = index;
                this._updateHighlight();
            }
        };

        thumb.appendChild(img);
        thumb.appendChild(label);
        thumb.appendChild(delBtn);

        const orderInput = document.createElement("input");
        orderInput.type = "number";
        orderInput.placeholder = String(index);
        const sortOrderValue = this.sortOrders[index];
        orderInput.value = (sortOrderValue !== undefined && sortOrderValue !== null) ? String(sortOrderValue) : "";

        orderInput.style.cssText = `
            width: 100%;
            padding: 2px 3px;
            background: #222;
            border-radius: 3px;
            border: 1px solid #444;
            color: #eee;
            font-size: 10px;
            box-sizing: border-box;
            text-align: center;
        `;

        orderInput.oninput = (e) => {
            const val = e.target.value.trim();
            if (val === "") {
                delete this.sortOrders[index];
            } else {
                const num = parseInt(val, 10);
                if (!isNaN(num)) {
                    this.sortOrders[index] = num;
                } else {
                    delete this.sortOrders[index];
                }
            }
        };

        wrapper.appendChild(thumb);
        wrapper.appendChild(orderInput);
        this.thumbnailElements.set(index, thumb);
        return wrapper;
    }

    _renderAllThumbnails() {
        if (this.renderScheduled) return;
        this.renderScheduled = true;

        requestAnimationFrame(() => {
            this.previewContainer.innerHTML = "";
            this.thumbnailElements.clear();
            const frag = document.createDocumentFragment();

            for (let i = 0; i < this.images.length; i++) {
                const url = buildImageUrl(this.images[i]);
                const dom = this._createThumbnail(i, url);
                frag.appendChild(dom);
            }

            this.previewContainer.appendChild(frag);

            const indexWidget = this.node.widgets?.find((w) => w.name === "index");
            if (indexWidget) {
                this.currentIndex = indexWidget.value ?? 0;
            }

            this._updateHighlight();
            this.renderScheduled = false;
        });
    }

    _updateHighlight() {
        if (this.renderScheduled) return;
        this.renderScheduled = true;

        requestAnimationFrame(() => {
            const active = this.currentIndex ?? 0;
            this.thumbnailElements.forEach((el, idx) => {
                if (idx === active) {
                    el.style.borderColor = "#0f0";
                    el.style.boxShadow = "0 0 6px #0f0";
                } else {
                    el.style.borderColor = "transparent";
                    el.style.boxShadow = "none";
                }
            });
            this.renderScheduled = false;
        });
    }

    applySorting() {
        if (!this.images.length) return;

        const orders = [];
        const wrappers = Array.from(this.previewContainer.children);

        wrappers.forEach((wrap, idx) => {
            const input = wrap.querySelector('input[type="number"]');
            if (!input) {
                orders[idx] = idx;
                return;
            }
            const raw = (input.value || "").trim();
            const num = raw === "" ? NaN : Number(raw);
            orders[idx] = Number.isFinite(num) ? num : idx;
            this.sortOrders[idx] = orders[idx];
        });

        const indexed = this.images.map((img, idx) => ({
            img,
            order: orders[idx] ?? idx,
        }));

        indexed.sort((a, b) => a.order - b.order);

        this.images = indexed.map((x) => x.img);
        this.sortOrders = {};

        this._syncWidgetFromImages();
        this._renderAllThumbnails();
    }

    _deleteImage(index) {
        if (index < 0 || index >= this.images.length) return;
        this.images.splice(index, 1);

        const newOrders = {};
        Object.keys(this.sortOrders).forEach((k) => {
            const oldIdx = parseInt(k);
            if (oldIdx < index) newOrders[oldIdx] = this.sortOrders[oldIdx];
            else if (oldIdx > index) newOrders[oldIdx - 1] = this.sortOrders[oldIdx];
        });
        this.sortOrders = newOrders;
        this._syncWidgetFromImages();
        this._renderAllThumbnails();
    }

    _clearAll() {
        this.initCancelled = true;
        this.images = [];
        this.sortOrders = {};
        this.thumbnailElements.clear();
        this.previewContainer.innerHTML = "";
        this._syncWidgetFromImages();
    }

    _showProgress(visible) {
        this.progressBar.style.display = visible ? "block" : "none";
        if (!visible) this.progressFill.style.width = "0%";
        else this._setProgress(0);
    }

    _setProgress(p) {
        const clamped = Math.max(0, Math.min(1, p));
        this.progressFill.style.width = `${clamped * 100}%`;
    }

    _sleep(ms) {
        return new Promise((res) => setTimeout(res, ms));
    }

    prepareForExecution() {
        this._syncWidgetFromImages();
    }

    prepareForSerialize() {
        this._syncWidgetFromImages();
    }

    destroy() {
        this.initCancelled = true;
        this.thumbnailElements.clear();
    }
}

function patchQueuePrompt() {
    const originalQueuePrompt = app.queuePrompt;
    if (originalQueuePrompt.__wanPatched) return;

    app.queuePrompt = async function (...args) {
        const nodes = app.graph?.findNodesByType("WanMultiImageLoader") || [];
        for (const node of nodes) {
            const c = node.__wanGallery;
            if (c) c.prepareForExecution();
        }
        return originalQueuePrompt.apply(this, args);
    };
    app.queuePrompt.__wanPatched = true;
}

app.registerExtension({
    name: "Comfy.WanMultiImageLoader",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WanMultiImageLoader") return;

        const origOnCreate = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const r = origOnCreate?.apply(this, arguments);
            let widget = this.widgets?.find((w) => w.name === "images_data");
            if (!widget) {
                widget = this.addWidget("text", "images_data", "[]", () => {}, {
                    multiline: false,
                });
            }
            widget.hidden = true;
            widget.serialize = true;

            this.__wanGallery = new WanMultiImageGallery(this, widget);

            this.addDOMWidget(
                "wan_multi_image_loader",
                "custom",
                this.__wanGallery.root,
            );

            this.setSize([420, 320]);

            const self = this;
            const origOnSerialize = this.onSerialize;
            this.onSerialize = function (o) {
                self.__wanGallery?.prepareForSerialize();
                origOnSerialize?.call(self, o);
            };

            const origOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                self.__wanGallery?.destroy();
                origOnRemoved?.call(self);
            };

            const origOnConfigure = this.onConfigure;
            this.onConfigure = function (o) {
                origOnConfigure?.apply(this, arguments);
                self.__wanGallery?._initializeFromWidget();
            };

            patchQueuePrompt();

            return r;
        };
    },
});
