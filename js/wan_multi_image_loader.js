import { app } from "../../scripts/app.js";

const onBeforeSerialize = LGraph.prototype.onBeforeSerialize;
LGraph.prototype.onBeforeSerialize = function() {
    onBeforeSerialize?.apply(this, arguments);
    
    const nodes = app.graph.findNodesByType("WanMultiImageLoader");
    for (const node of nodes) {
        if (node.syncDataIfDirty) {
            node.syncDataIfDirty();
        }
    }
};

app.registerExtension({
    name: "Comfy.WanMultiImageLoader",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WanMultiImageLoader") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                const node = this;
                
                let imagesDataWidget = this.widgets.find(w => w.name === "images_data");
                if (!imagesDataWidget) {
                    imagesDataWidget = this.addWidget("text", "images_data", "[]", () => {}, {
                        serialize: true
                    });
                    imagesDataWidget.type = "hidden";
                    imagesDataWidget.computeSize = () => [0, -4];
                }
                
                const container = document.createElement("div");
                container.style.cssText = "width:100%;padding:8px;background:#1a1a1a;border-radius:6px;margin:5px 0;";
                const btnContainer = document.createElement("div");
                btnContainer.style.cssText = "display:flex;gap:6px;margin-bottom:8px;";
                const uploadBtn = document.createElement("button");
                uploadBtn.textContent = "📁 选择";
                uploadBtn.style.cssText = "flex:1;padding:8px;background:#2a2a2a;color:#fff;border:1px solid #444;border-radius:4px;cursor:pointer;font-size:13px;";
                const addBtn = document.createElement("button");
                addBtn.textContent = "➕ 增加";
                addBtn.style.cssText = "flex:1;padding:8px;background:#2a4a2a;color:#fff;border:1px solid #4a6;border-radius:4px;cursor:pointer;font-size:13px;";
                const sortBtn = document.createElement("button");
                sortBtn.textContent = "🔃 排序";
                sortBtn.style.cssText = "padding:8px;background:#2a2a4a;color:#fff;border:1px solid #46a;border-radius:4px;cursor:pointer;font-size:13px;";
                const clearBtn = document.createElement("button");
                clearBtn.textContent = "🗑️";
                clearBtn.style.cssText = "padding:8px;background:#4a2a2a;color:#fff;border:1px solid #a44;border-radius:4px;cursor:pointer;font-size:13px;";
                const fileInput = document.createElement("input");
                fileInput.type = "file";
                fileInput.multiple = true;
                fileInput.accept = "image/*";
                fileInput.style.display = "none";
                const progressBar = document.createElement("div");
                progressBar.style.cssText = "display:none;height:3px;background:#333;border-radius:2px;margin-bottom:6px;overflow:hidden;";
                const progressFill = document.createElement("div");
                progressFill.style.cssText = "height:100%;background:#4a6;width:0%;transition:width 0.2s;";
                progressBar.appendChild(progressFill);
                const previewContainer = document.createElement("div");
                previewContainer.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:6px;max-height:300px;overflow-y:auto;background:#252525;padding:6px;border-radius:4px;";
                
                let images = []; 
                let sortOrders = {};
                let isAdding = false;
                let isDirty = false;
                
                const compressImage = (base64Data) => {
                    return new Promise((resolve) => {
                        const img = new Image();
                        img.onload = () => {
                            const canvas = document.createElement('canvas');
                            const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
                            canvas.width = 128;
                            canvas.height = 128;
                            const scale = Math.min(128 / img.width, 128 / img.height);
                            const w = Math.floor(img.width * scale);
                            const h = Math.floor(img.height * scale);
                            const x = Math.floor((128 - w) / 2);
                            const y = Math.floor((128 - h) / 2);
                            ctx.fillStyle = '#000';
                            ctx.fillRect(0, 0, 128, 128);
                            ctx.drawImage(img, x, y, w, h);
                            resolve(canvas.toDataURL('image/jpeg', 0.3));
                        };
                        img.onerror = () => resolve(null); 
                        img.src = base64Data;
                    });
                };

                const syncWidgetData = () => {
                    const dataToStore = images.map(item => ({ name: item.name, data: item.data }));
                    imagesDataWidget.value = JSON.stringify(dataToStore);
                    isDirty = false; 
                };

                this.syncDataIfDirty = () => {
                    if (isDirty) {
                        syncWidgetData();
                    }
                };

                const updateHighlight = () => {
                    const indexWidget = node.widgets.find(w => w.name === "index");
                    const currentIndex = indexWidget ? indexWidget.value : 0;
                    previewContainer.querySelectorAll("div[data-index]").forEach(container => {
                        const thumb = container.querySelector("div");
                        const idx = parseInt(container.dataset.index);
                        if (idx === currentIndex) {
                            thumb.style.borderColor = "#0f0";
                            thumb.style.boxShadow = "0 0 8px #0f0";
                        } else {
                            thumb.style.borderColor = "transparent";
                            thumb.style.boxShadow = "none";
                        }
                    });
                };

                const createThumbnail = (index, compressedData, parent) => {
                    const container = document.createElement("div"); container.dataset.index = index; container.style.cssText = "display:flex;flex-direction:column;gap:3px;";
                    const thumb = document.createElement("div"); thumb.style.cssText = "position:relative;aspect-ratio:1;border-radius:4px;overflow:hidden;cursor:pointer;border:2px solid transparent;background:#000;";
                    const img = document.createElement("img"); img.src = compressedData; img.style.cssText = "width:100%;height:100%;object-fit:cover;";
                    const label = document.createElement("div"); label.textContent = `#${index}`; label.style.cssText = "position:absolute;top:2px;left:2px;background:rgba(0,0,0,0.7);color:#fff;padding:2px 4px;border-radius:3px;font-size:11px;pointer-events:none;";
                    const deleteBtn = document.createElement("button"); deleteBtn.textContent = "×"; deleteBtn.style.cssText = "position:absolute;top:2px;right:2px;width:20px;height:20px;background:rgba(255,0,0,0.7);color:#fff;border:none;border-radius:3px;cursor:pointer;font-size:16px;line-height:1;";
                    
                    deleteBtn.onclick = (e) => {
                        e.stopPropagation();
                        images.splice(index, 1);
                        
                        delete sortOrders[index];
                        const newOrders = {};
                        Object.keys(sortOrders).forEach(k => {
                            const oldIdx = parseInt(k);
                            if (oldIdx > index) newOrders[oldIdx - 1] = sortOrders[k];
                            else if (oldIdx < index) newOrders[oldIdx] = sortOrders[k];
                        });
                        sortOrders = newOrders;
                        
                        redrawAll();
                        isDirty = true;
                    };
                    
                    thumb.appendChild(img); thumb.appendChild(label); thumb.appendChild(deleteBtn);
                    
                    thumb.onclick = () => {
                        const indexWidget = node.widgets.find(w => w.name === "index");
                        if (indexWidget) {
                            indexWidget.value = index;
                            updateHighlight();
                        }
                    };
                    
                    const orderInput = document.createElement("input"); orderInput.type = "number"; orderInput.placeholder = index; orderInput.value = sortOrders[index] !== undefined ? sortOrders[index] : ""; orderInput.style.cssText = "width:100%;padding:3px;background:#2a2a2a;border:1px solid #444;border-radius:3px;color:#fff;font-size:11px;text-align:center;";
                    orderInput.oninput = (e) => {
                        const val = e.target.value.trim();
                        sortOrders[index] = val !== "" ? parseInt(val) : undefined;
                    };
                    
                    container.appendChild(thumb); container.appendChild(orderInput); parent.appendChild(container);
                };

                const redrawAll = () => {
                    previewContainer.innerHTML = "";
                    const fragment = document.createDocumentFragment();
                    
                    for (let i = 0; i < images.length; i++) {
                        createThumbnail(i, images[i].thumb, fragment);
                    }
                    
                    previewContainer.appendChild(fragment);
                    updateHighlight();
                };

                const applySorting = () => {
                    const indices = [];
                    for (let i = 0; i < images.length; i++) {
                        indices.push({
                            index: i,
                            order: sortOrders[i] !== undefined ? sortOrders[i] : 999 + i
                        });
                    }
                    indices.sort((a, b) => a.order - b.order);
                    
                    const sorted = indices.map(item => images[item.index]);
                    images = sorted; 
                    sortOrders = {};
                    
                    redrawAll();
                    isDirty = true;
                };
                
                const processFiles = async (files, replace = true) => {
                    if (replace) {
                        images = []; 
                        sortOrders = {};
                        previewContainer.innerHTML = "";
                    }
                    
                    const startIndex = images.length;
                    const totalFiles = files.length;
                    
                    progressBar.style.display = "block";
                    uploadBtn.disabled = true;
                    addBtn.disabled = true;
                    
                    const batchSize = 5;
                    
                    for (let i = 0; i < totalFiles; i += batchSize) {
                        const batch = files.slice(i, Math.min(i + batchSize, totalFiles));
                        const batchFragment = document.createDocumentFragment();

                        await Promise.all(batch.map(async (file, batchIdx) => {
                            const actualIndex = startIndex + i + batchIdx;
                            return new Promise((resolve) => {
                                const reader = new FileReader();
                                reader.onload = async (event) => {
                                    const base64Data = event.target.result;
                                    const compressedData = await compressImage(base64Data);
                                    images[actualIndex] = { name: file.name, data: base64Data, thumb: compressedData };
                                    createThumbnail(actualIndex, compressedData, batchFragment); 
                                    resolve();
                                };
                                reader.readAsDataURL(file);
                            });
                        }));
                        
                        previewContainer.appendChild(batchFragment); 
                        progressFill.style.width = Math.min(((i + batchSize) / totalFiles) * 100, 100) + "%";
                        await new Promise(resolve => setTimeout(resolve, 0));
                    }
                    
                    syncWidgetData(); 
                    
                    setTimeout(() => {
                        progressBar.style.display = "none";
                        progressFill.style.width = "0%";
                        uploadBtn.disabled = false;
                        addBtn.disabled = false;
                        updateHighlight();
                    }, 200);
                };

                const initializeFromWidget = async () => {
                    try {
                        const data = JSON.parse(imagesDataWidget.value);
                        if (!data || data.length === 0) {
                            images = [];
                            return;
                        }
                        
                        images = []; 
                        previewContainer.innerHTML = "";
                        progressBar.style.display = "block";
                        progressFill.style.width = "0%";
                        const total = data.length;
                        const fragment = document.createDocumentFragment();

                        for (let i = 0; i < total; i++) {
                            const item = data[i];
                            const compressedData = await compressImage(item.data); 
                            images[i] = { name: item.name, data: item.data, thumb: compressedData };
                            createThumbnail(i, compressedData, fragment);
                            
                            if (i % 10 === 0 || i === total - 1) {
                                progressFill.style.width = Math.min(((i + 1) / total) * 100, 100) + "%";
                                await new Promise(resolve => setTimeout(resolve, 0));
                            }
                        }
                        
                        previewContainer.appendChild(fragment);
                        updateHighlight();

                    } catch (e) {
                        console.error("Error initializing WanMultiImageLoader:", e);
                        images = [];
                        imagesDataWidget.value = "[]";
                    } finally {
                        isDirty = false; 
                        setTimeout(() => {
                            progressBar.style.display = "none";
                            progressFill.style.width = "0%";
                        }, 200);
                    }
                };
                
                fileInput.onchange = async (e) => {
                    const files = Array.from(e.target.files);
                    if (files.length > 0) {
                        await processFiles(files, !isAdding);
                        fileInput.value = "";
                        isAdding = false;
                    }
                };
                
                const indexWidget = this.widgets.find(w => w.name === "index");
                if (indexWidget) {
                    const originalCallback = indexWidget.callback;
                    indexWidget.callback = function() {
                        originalCallback?.apply(this, arguments);
                        updateHighlight();
                    };
                }
                
                uploadBtn.onclick = () => { isAdding = false; fileInput.click(); };
                addBtn.onclick = () => { isAdding = true; fileInput.click(); };
                sortBtn.onclick = () => applySorting();
                
                clearBtn.onclick = () => {
                    if (confirm("确定清空？")) {
                        images = [];
                        sortOrders = {};
                        previewContainer.innerHTML = "";
                        syncWidgetData(); 
                    }
                };
                
                btnContainer.appendChild(uploadBtn); btnContainer.appendChild(addBtn); btnContainer.appendChild(sortBtn); btnContainer.appendChild(clearBtn);
                container.appendChild(btnContainer); container.appendChild(progressBar); container.appendChild(previewContainer); container.appendChild(fileInput);
                
                this.addDOMWidget("multi_image_loader", "customwidget", container);
                this.setSize([400, 280]);
                
                initializeFromWidget();
                
                return result;
            };
        }
    }
});
