
// -----------------------------------------------------------------------------
// Standard formats
// -----------------------------------------------------------------------------
var fmt_pct_1_1 = new Intl.NumberFormat('en-US',
						{
						style: "percent",
						minimumFractionDigits:1,
						maximumFractionDigits:1
						});
var fmt_pct_3_1 = new Intl.NumberFormat('en-US',
						{
						style: "percent",
						minimumIntegerDigits:3,
						minimumFractionDigits:1,
						maximumFractionDigits:1
						});
	// -----------------------------------------------------------------------------------------------------
	// Function definitions
	// -----------------------------------------------------------------------------------------------------
	initial_layout = function() {

		// Create items
		var item_list=[];
		var item_list_icons=[];
		var item_content_divs=[];


		var contador_ol=1;
		var var_li_status='';
		var var_li2_status='';
		for (var i = 0; i < json_items.length; i++) {
			json_items[i]=JSON.parse(json_items[i]);
			var item_class ='';
			switch (json_items[i].type) {
				case 'preprocessing': item_class=' class="icon_cog" ';break;
				case 'univariate': item_class=' class="icon_chart" ';break;
				case 'prebinning': item_class=' class="icon_binning" ';break;
				case 'toymodelselection': item_class=' class="icon_model" ';break;
				default: item_class=' class="icon_default" ';break;
			}

			var href_id=json_items[i].item_id;
			if (json_items[i].step_id == '00' && json_items[i].item_info==0) {
				href_id+='_01';
			}
			else {
				href_id+='_'+json_items[i].step_id;
				item_content_divs.push('<div class="cont_item" style="display:none;" id="'+href_id+'"></div>');
			}

			if (json_items[i].step_id == '00') {
				if (var_li2_status=='open') {
					item_list.push('</ol>');
					item_list_icons.push('</ol>');
					var_li2_status='closed';
				}
				if (var_li_status=='open') {
					item_list.push('</li>');
					var_li_status='closed';
				}
				item_list.push('<li>');
				item_list.push('<a '+item_class+' href="#' + href_id + '">');
				item_list.push(  json_items[i].title + "</a>");
				item_list_icons.push('<li>');
				item_list_icons.push('<a '+item_class+' href="#' + href_id + '"></a>');
			}
			else {
				if (var_li2_status!='open') {
					item_list.push('<ol>');
					item_list_icons.push('<ol>');
					var_li2_status='open';
					contador_ol=1;
				}
				item_list.push('<li class="sidebar_subitem">');
				item_list.push('<a '+' href="#' + href_id + '">');
				item_list.push( contador_ol +'. '+ json_items[i].title + "</a>");
				item_list.push('</li>');
				item_list_icons.push('<li class="sidebar_subitem">');
				item_list_icons.push('<a '+' href="#' + href_id + '">');
				item_list_icons.push( contador_ol +''+ '' + "</a>");
				item_list_icons.push('</li>');
				contador_ol+=1;
			}
		}
		$('#toc').append($(item_list.join('')));
		$('#toc_icons').append($(item_list_icons.join('')));
		$('#main_content').append($(item_content_divs.join('')));

		// Titles
		var titles_html=[];
		report_info=JSON.parse(report_info);
		titles_html.push('<div id="title">');
		titles_html.push('<h1>'+report_info['title1']+'</h1>');
		titles_html.push('<h2>'+report_info['title2']+'</h2>');
		titles_html.push('<h3>'+report_info['title3']+'</h3>');
		$('#main').append($(titles_html.join('')));

		// Particles
		particlesJS("particles-js", {"particles":{"number":{"value":400,"density":{"enable":true,"value_area":800}},"color":{"value":"#ffffff"},"shape":{"type":"circle","stroke":{"width":0,"color":"#000000"},"polygon":{"nb_sides":5},"image":{"src":"img/github.svg","width":100,"height":100}},"opacity":{"value":0.3,"random":false,"anim":{"enable":false,"speed":1,"opacity_min":0.1,"sync":false}},"size":{"value":2,"random":true,"anim":{"enable":false,"speed":40,"size_min":0.1,"sync":false}},"line_linked":{"enable":true,"distance":110,"color":"#ffffff","opacity":0.05,"width":1},"move":{"enable":true,"speed":2.0,"direction":"none","random":false,"straight":false,"out_mode":"out","bounce":false,"attract":{"enable":false,"rotateX":600,"rotateY":1200}}},"interactivity":{"detect_on":"canvas","events":{"onhover":{"enable":false,"mode":"repulse"},"onclick":{"enable":false,"mode":"push"},"resize":true},"modes":{"grab":{"distance":400,"line_linked":{"opacity":1}},"bubble":{"distance":400,"size":40,"duration":2,"opacity":8,"speed":3},"repulse":{"distance":200,"duration":0.4},"push":{"particles_nb":4},"remove":{"particles_nb":2}}},"retina_detect":true});
		window.pJSDom[0].pJS.particles.move.enable = false;
	}

	// -----------------------------------------------------------------------------------------------------
	render_step_list = function(item) {
		var html='';

		if (item.data.steps.length>1) {
			html+='<div class="pos_step_list"><ul class="step_list">';
			for (var i = 0; i < item.data.steps.length; i++) {
				html+='<li><a href="#'+item.data.id+'_'+item.data.steps[i].step_id+'">' + item.data.steps[i].name + '</a></li>';;
			}
			for (var i = 0; i < item.data.steps.length; i++) {
				html+='<li><a href="#'+item.data.id+'_'+item.data.steps[i].step_id+'">' + item.data.steps[i].name + '</a></li>';;
			}
			html+='</ul></div>';
		}
		return html;
	}

	// -----------------------------------------------------------------------------------------------------
	create_step_layout_parts = function(item) {
		var layout={};
		if (item.json_data.item_layout_version == '01') {
			layout['header'] = document.createElement('div');
			layout['header'].setAttribute('class','pos_header');
			layout['sidebar_left']= document.createElement('div',{class: "pos_sidebar_left"});
			layout['sidebar_left'].setAttribute('class','pos_sidebar_left');
			layout['sidebar_right'] = document.createElement('div',{"class": "pos_sidebar_right"});
			layout['sidebar_right'].setAttribute('class','pos_sidebar_right');
			layout['mainbody'] = document.createElement('div',{class: "pos_mainbody"});
			layout['mainbody'].setAttribute('class','pos_mainbody');
			layout['mainbody_top'] = document.createElement('div',{class: "pos_mainbody_top"});
			layout['mainbody_top'].setAttribute('class','pos_mainbody_top');
			layout['mainbody_middle']= document.createElement('div',{class: "pos_mainbody_middle"});
			layout['mainbody_middle'].setAttribute('class','pos_mainbody_middle');
			layout['mainbody_bottom'] = document.createElement('div',{class: "pos_mainbody_bottom"});
			layout['mainbody_bottom'].setAttribute('class','pos_mainbody_bottom');
			layout['footer'] = document.createElement('div',{class: "pos_footer"});
			layout['footer'].setAttribute('class','pos_footer');
		}
		return layout;
	}
	// -----------------------------------------------------------------------------------------------------
	join_step_layout_parts = function(item,step_layout_parts) {
		var step_layout = document.createDocumentFragment();
		if (item.json_data.item_layout_version == '01') {
			step_layout.appendChild(step_layout_parts['header']);
			step_layout.appendChild(step_layout_parts['sidebar_left']);
			step_layout_parts['mainbody'].appendChild(step_layout_parts['mainbody_top']);
			step_layout_parts['mainbody'].appendChild(step_layout_parts['mainbody_middle']);
			step_layout_parts['mainbody'].appendChild(step_layout_parts['mainbody_bottom']);
			step_layout.appendChild(step_layout_parts['mainbody']);
			step_layout.appendChild(step_layout_parts['sidebar_right']);
			step_layout.appendChild(step_layout_parts['footer']);
		}
		return step_layout;
	}
	// -----------------------------------------------------------------------------------------------------
	render_element_basic = function(item,step,element) {
		var element_fragment = document.createElement('div');
		element_fragment.innerHTML=JSON.stringify(element.data);
		return element_fragment;
	}


// -----------------------------------------------------------------------------------------------------
// Column renderers;
// -----------------------------------------------------------------------------------------------------
	render_col_action = function(row,pos) {
		var val_class='';
		switch (row[pos]) {
			case 'keep': val_class=' class='+'act_ok';break;
			case 'remove': val_class=' class='+'act_rm';break;
			case 'transform': val_class=' class='+'act_tr';break;
			case 'repair': val_class=' class='+'act_tr';break;
		}
		return '<td'+val_class+'>'+row[pos]+'</td>'
	}

	render_col_check = function(row,pos,action) {
		var main_class='';
		var color_class='';
		var cell_content='';
		switch (row[pos]) {
			case 1:
				cell_content='&#x25cf;';
				main_class='chk';
				switch (row[action]) {
					case 'keep': color_class='color_ok';break;
					case 'remove': color_class='color_rm';break;
					case 'transform': color_class='color_tr';break;
					case 'repair': color_class='color_tr';break;
				}
				break;
			default: cell_content='·';
		}
		return '<td class="'+color_class+' '+main_class+'">'+cell_content+'</td>';
	}

	render_col_block_number = function(row,pos,renderer) {
		switch (row[pos]) {
			case -1: return '<td></td>';
		}
		return '<td>'+renderer.format_number.format(row[pos])+'</td>';
	}

	render_col_raw = function(row,pos) {
		return '<td>'+row[pos]+'</td>'
	}

	render_col_comment = function(row,pos,pos_color,main_style) {
		switch (row[pos_color]) {
			case 'keep': val_class=' class='+'color_ok';break;
			case 'remove': val_class=' class='+'color_rm';break;
			case 'transform': val_class=' class='+'color_tr';break;
			case 'repair': val_class=' class='+'color_tr';break;
		}
		if (row[pos]=='') return '<td style="'+main_style+'"'+'></td>';
		return '<td style="'+main_style+'"'+val_class+'><span data-tocoltip="'+row[pos]+'">'+'<span class="dtc_icon_warning">'+row[pos]+'</span></span></td>'
	}

	render_col_alert_tooltip = function(row,pos,pos_color,main_style) {
		switch (row[pos_color]) {
			case 'keep': val_class=' class='+'color_ok';break;
			case 'remove': val_class=' class='+'color_rm';break;
			case 'transform': val_class=' class='+'color_tr';break;
			case 'repair': val_class=' class='+'color_tr';break;
		}
		if (row[pos]=='') return '<td style="'+main_style+'"'+'></td>';
		//return '<td style="'+main_style+'"'+val_class+'><span data-tooltip="'+row[pos]+'">'+'<span class="dt_icon_warning">'+row[pos]+'</span></span></td>'
		return '<td style="'+main_style+'"'+val_class+'><div class="tooltip dt_icon_warning"><span class="tooltiptext">'+row[pos]+'</span></div></td>'
		return '<td style="'+main_style+'"'+val_class+'><div class="tooltip">TOOLTIP<span class="tooltiptext">'+row[pos]+'</span></div></td>'
		return '<td style="'+main_style+'"'+val_class+'><div class="tooltip dt_icon_warning"><span class="tooltiptext">'+row[pos]+'</span></div></td>'
	}

	render_col_case = function(row,pos,pos_color,main_style) {
		switch (row[pos_color]) {
			case 'keep': val_class=' class='+'color_ok';break;
			case 'remove': val_class=' class='+'color_rm';break;
			case 'transform': val_class=' class='+'color_tr';break;
			case 'repair': val_class=' class='+'color_tr';break;
		}
		if (row[pos]=='') return '<td style="'+main_style+'"'+'></td>';
		return '<td style="'+main_style+'"'+val_class+'>'+row[pos]+''+'</span></td>'
	}

	render_col_linked_variable = function(row,pos,row_number) {
		var item_id = render_col_linked_variable.caller.caller.arguments[0].json_data.item_id;
		var step_pos = render_col_linked_variable.caller.caller.arguments[1];
		var step_id = render_col_linked_variable.caller.caller.arguments[0].json_data.item_steps[step_pos].step_id;
		var content_pos = render_col_linked_variable.caller.caller.arguments[2];
		var block_pos = render_col_linked_variable.caller.caller.arguments[3];
		var details_id = item_id +'_'+step_id+'_'+content_pos+'_'+block_pos+'_'+row[pos];
		return '<td><button row_pos="'+row_number+'" id="'+ details_id +'" class="modal_button icon_link">'+row[pos]+'</button></td>'
	}

	render_col_numeric = function(row,pos,renderer) {
		return '<td '+renderer.class+'>'+renderer.format_number.format(row[pos])+'</td>'
	}

	render_col_numeric_num_pct = function(row,pos,renderer,summary_columns_positions) {
		html='<td class="left">';
		html+='<span style="display: none;">'+fmt_pct_3_1.format(row[summary_columns_positions[renderer.pct_column]])+'</span>';
		html+='<span class="num_pct_left">';
		html+=fmt_pct_1_1.format(row[summary_columns_positions[renderer.pct_column]]);
		html+='</span>'
		html+='<span class="value_small">';;
		html+='('+renderer.format_number.format(row[pos])+')';
		html+='</span>';
		html+= '</td>';
		return html;
	}

	render_col_elements = function(row,pos,style='',max,renderer) {
		var html = '';
		var num_elements=row[pos];

		html+='<td class="" style="'+style+'">';
		if (max!=1) {
			html+='<span style="display: none;">'+renderer.format_number.format(row[pos])+'</span>';
			html+='<span style="display: inline-block;width:15px;text-align:right;margin-right:5px;">'+num_elements+'</span>';
			html+='<span class="element_filled">'+"&#9611;".repeat(Math.min(row[pos],max))+'</span>';
			html+='<span class="element_empty">'+"&#9611;".repeat(max-Math.min(row[pos],max))+'</span>';
		}
		else {
			html+='<span style="display: none;">'+row[pos]+'</span>';
			if (row[pos]==1) html+='<span class="element_filled white">&#9611;</span>';
			else html+='<span class="element_empty">&#9611;</span>';
		}
		html+='</td>';
		return html;
	}
	// -----------------------------------------------------------------------------------------------------
	render_col_iv = function(row,pos,style,action,renderer) {
		var html = '';
		var val_class='bar';
		switch (row[action]) {
			case 'remove': val_class+=' err';break;
		}
		html='<td class="left">';
		if (row[pos]>1.1) {
			html+='<span class="'+val_class+'" style="margin: 6px 1px;width:'+ Math.min(55,eval((row[pos]*55)+2)) +'px"></span>';
			html+='<span class="'+val_class+'" style="width:1px;margin: 6px 1px;"></span>';
			html+='<span class="'+val_class+'" style="width:1px;margin: 6px 1px 6px 0px;"></span>';
			html+='<span class="'+val_class+'" style="width:10px;margin: 6px 4px 6px 1px;"></span>';
		}
		else {
			html+='<span class="'+val_class+'" style="width:'+ Math.min(65,eval((row[pos]*65)+2)) +'px"></span>';
		}
		html+=renderer.format_number.format(row[pos]);
		html+='</td>';
		return html;
	}
	// -----------------------------------------------------------------------------------------------------
	render_col_trend = function(row,pos,groups,variable_type,trend_changes=0) {
/*
		if (row[variable_type] == 'nominal' || row[variable_type] == 'categorical' || row[groups]==1) {
			return '<td></td>';
		}
*/

		var val_class='';
		var val=row[pos];
		switch (row[pos]) {
			case 'ascending':
				val_class=' class="trend_flat_asc"';
				break;
			case 'descending':
				val_class=' class="trend_flat_desc"';
				break;
			case 'undefined':
				val='-';
				break;

			//case 'concave': val_class=' class='+'trend_concave';break;
			//case 'convex': val_class=' class='+'trend_convex';break;
		}
		return '<td'+val_class+'>'+val+'</td>';
	}

// -----------------------------------------------------------------------------
// Column headeers & renderers;
// -----------------------------------------------------------------------------
	get_col_header = function(col_name) {
		var text = col_name;
		var style = '';
		switch (col_name.toLowerCase()) {
			case 'group_missing': 		text='M';style='min-width:12px;';break;
			case 'group_others': 		text='O';style='min-width:12px;';break;
			case 'group_special': 		text='S';style='min-width:12px;';break;
			case 'largest_bucket_(%)': 	text='largest';break;
			case 'smallest_bucket_(%)': text='smallest';break;
			case 'iv': 					style='min-width:105px;';break;
		}
		return '<th style="'+style+'">'+text+'</th>';
	}

	assing_col_render = function(item_type,step_type,col_name) {
		var renderer = new Object();
		renderer.class='';
		renderer.format_number = new Intl.NumberFormat('en-US');
		var name ='';
		if (item_type=='preprocessing') {
			switch (col_name) {
				case 'empty': 			name='check';break;
				case 'constant': 		name='check';break;
				case 'nan/unique': 		name='check';break;
				case 'special/unique': 	name='check';break;
				case 'id': 				name='check';break;
				case 'dates': 			name='check';break;
				case 'binary': 			name='check';break;
				case 'target': 			name='check';break;
				case 'special': 		name='check';break;
				case 'numeric_string': 	name='check';break;
				case 'duplicated': 		name='check';break;
				case 'duplicated_of': 	name='raw';break;
				case 'block': 			name='block_number';
					renderer.format_number = new Intl.NumberFormat('en-US',
					{
						minimumIntegerDigits:2
					});
					break;
			}
		}
		else if (item_type=='univariate') {
			if (step_type=='analysis') {
				switch (col_name) {
					case 'variable': 		name='linked_variable';break;
					case 'unique_specials':	name='none';break;
					case 'stability_1': 	name='numeric';
					case 'stability_2': 	name='numeric';
					case 'stability_info_1':name='numeric';
					case 'stability_info_2':name='numeric';
					case 'concentration': 	name='numeric';
					case 'mean': 			name='numeric';
					case 'std': 			name='numeric';
						renderer.format_number = new Intl.NumberFormat('en-US',
						{
							minimumFractionDigits:3,
							maximumFractionDigits:3
						});
						break;
				}
			}
		}
		else if (item_type=='optimalgrouping') {
			if (step_type=='analysis') {
				switch (col_name) {
					case 'variable': 			name='linked_variable';break;
					case 'pd_monotonicity':		name='trend';break;
					case 'groups': 				name='elements_100px';
						renderer.format_number = new Intl.NumberFormat('en-US',
						{
							minimumIntegerDigits:2
						});
						break;
					case 'group_missing': 		name='elements';break;
					case 'group_others': 		name='elements';break;
					case 'group_special': 		name='elements';break;
					case 'max_p_value':
						name='numeric';
						renderer.format_number = new Intl.NumberFormat('en-US',
						{
							minimumFractionDigits:3,
							maximumFractionDigits:3
						});
						break;

					case 'largest_bucket':
						name='numeric_num_pct';
						renderer.pct_column='largest_bucket_(%)';
						break;
					case 'largest_bucket_(%)':name='none';break;

					case 'smallest_bucket':
						name='numeric_num_pct';
						renderer.pct_column='smallest_bucket_(%)';
						break;
					case 'smallest_bucket_(%)':name='none';break;

					case 'std_buckets':
						name='numeric';
						renderer.class='class="right"';
						renderer.format_number = new Intl.NumberFormat('en-US',
						{
							maximumFractionDigits:0
						});
						break;
				}
			}
		}

		if (step_type=='analysis') {
			switch (col_name) {
				case 'comment': 	name='alert_tooltip';break;
			}
		}

		// Default
		if (name=='') {
			switch (col_name) {
				case 'variable': 	name='raw';break;
				case 'action': 		name='action';break;
				case 'comment': 	name='comment';break;
				case 'case': 		name='case';break;
				case 'iv': 			name='iv';
					renderer.format_number = new Intl.NumberFormat('en-US',
					{
						minimumFractionDigits:3,
						maximumFractionDigits:3
					});
					break;
				default:  			name='raw';break;
			}
		}
		renderer.name=name;
		return renderer;
	}

	apply_col_render = function(renderer,row,pos,summary_columns_positions,row_number) {
		switch (renderer.name) {
			case 'linked_variable': return render_col_linked_variable(row,pos,row_number);
			case 'action': 	return render_col_action(row,pos);
			case 'comment': 	return render_col_case(row,pos,summary_columns_positions['action'],"width:200px;");
			case 'alert_tooltip': 	return render_col_alert_tooltip(row,pos,summary_columns_positions['action'],"width:47px;");
			case 'case': 		return render_col_case(row,pos,summary_columns_positions['action'],"width:50px;text-align:left;");
			case 'check': 	return render_col_check(row,pos,summary_columns_positions['action']);
			case 'block_number': 	return render_col_block_number(row,pos,renderer);
			case 'stability': 	return render_col_stability(row,pos);
			case 'raw': 	return render_col_raw(row,pos);
			case 'iv': 	return render_col_iv(row,pos,'',summary_columns_positions['action'],renderer);
			case 'trend': 	return render_col_trend(row,pos,summary_columns_positions['groups'],summary_columns_positions['type'],summary_columns_positions['trend changes']);
			case 'elements': 	return render_col_elements(row,pos,'',1);
			case 'elements_100px': 	return render_col_elements(row,pos,'',10,renderer);
			case 'numeric': 		return render_col_numeric(row,pos,renderer);
			case 'numeric_num_pct': 		return render_col_numeric_num_pct(row,pos,renderer,summary_columns_positions);
			case 'none': 		return '';
			default:  		return render_col_raw(row,pos);
		}
	}

// -----------------------------------------------------------------------------------------------------
// Assign block renderers;
// -----------------------------------------------------------------------------------------------------

	assing_block_render = function(item_type,step_type,content_type,block_type) {
		switch (block_type) {
				case 'db_info': 		return 'big_number';
				case 'db_info_expanded': return 'big_number_2';
				case 'column_analysis': return 'table';
				case 'results': 		return 'table';
				case 'block_analysis': 	return 'table';
				case 'config': 			return 'config';
				case 'cpu_time': 		return 'piechart';
				case 'table': 			return 'summary';
				case 'table_num_ord': 	return 'summary';
				case 'table_cat_nom': 	return 'summary';
				default:  				return 'raw';
		}
	}

// -----------------------------------------------------------------------------------------------------
// Apply block renderers;
// -----------------------------------------------------------------------------------------------------

	apply_block_render = function(block_renderer,item,step_pos,content_pos,block_pos) {
		switch (block_renderer) {
			case 'big_number': 		return render_block_big_number(item,step_pos,content_pos,block_pos);
			case 'big_number_2': 	return render_block_big_number(item,step_pos,content_pos,block_pos,2);
			case 'table': 			return render_block_table(item,step_pos,content_pos,block_pos);
			case 'piechart': 		return render_block_piechart(item,step_pos,content_pos,block_pos);
			case 'summary': 		return render_block_summary(item,step_pos,content_pos,block_pos);
			case 'config': 		return render_block_config(item,step_pos,content_pos,block_pos);
			default:  				return render_block_raw(item,step_pos,content_pos,block_pos);
		}
	}
	// -----------------------------------------------------------------------------------------------------
	get_block_description = function(block_name) {
		switch (block_name) {
			case 'column_analysis': return 'Column analysis';
			case 'block_analysis': 	return 'Block analysis';
			case 'db_info': 		return 'Database information';
			case 'db_info_expanded':return 'Database information';
			case 'config': 			return 'Configuration options';
			case 'cpu_time': 		return 'CPU time';
			case 'results': 		return 'Results';
			case 'table': 			return 'Summary table';
			case 'table_num_ord': 	return 'Summary table';
			case 'table_cat_nom': 	return 'Summary table';
			default:  				return block_name;
		}
	}
	// -----------------------------------------------------------------------------------------------------
	get_block_subdescription = function(block_name) {
		switch (block_name) {
			case 'table_num_ord': 	return 'Numeric & Ordinal variables';
			case 'table_cat_nom': 	return 'Categorical & Nominal variables';
			default:  				return '';
		}
	}



	// -----------------------------------------------------------------------------------------------------
	// 2018/06/30
	render_block_summary = function(item,step_pos,content_pos,block_pos,format='html') {
		console.time('___render_block_summary()');
		var step = item.json_data.item_steps[step_pos];
		var element = step.step_contents[content_pos];
		var block = element.content_blocks[block_pos];

		var num_columns=block.block_data.columns.length;
		var num_rows=block.block_data.data.length;

		var datatable = new Object();
		var indexed_list = new Object();
		var indexed_desc_list = new Object();

		// detect columns;
		var summary_columns= [];
		for (var i = 0; i < num_columns; i++) {
			var summary_column= new Object();
			summary_column.pos=i;
			summary_column.name=block.block_data.columns[i].toLowerCase();
			summary_column.renderer=assing_col_render(item.json_data.item_type,step.step_type,summary_column.name);
			summary_columns.push(summary_column);
		}
		var summary_columns_positions= new Object();
		for (var i = 0; i < num_columns; i++) {
			summary_columns_positions[summary_columns[i].name]=i;
		}

		datatable_id='summary_'+step_pos+'_'+content_pos+'_'+block_pos+'_'+Date.now();

		var html='';
		html+='<div class="block_summary">';
		html+='<h1 class="block_title">'+get_block_description(block.block_type);
		var sub_description = get_block_subdescription(block.block_type);
		if (sub_description!='') {
			html+=': <span class="block_title_extra">'+sub_description+'</span>';
		}
		html+='</h1>';

		html+='<table id="'+datatable_id+'" class="summary_table">';
		html+='<thead><tr>';
		for (var i = 0; i < num_columns; i++) {
			// Original column name, not lowercase
			var col_name = block.block_data.columns[i];
			var col_renderer = summary_columns[i].renderer;
			if (col_renderer.name != 'none') {
				html+=get_col_header(col_name);
			}
			// Linked_variable --> Generate indexed list of rows for
			//   navigation purposes (modal boxes)
			if (col_renderer.name == 'linked_variable') {
				for (var j = 0; j <= num_rows-1; j++) {
					indexed_list[j]= item.json_data.item_id +'_'+step.step_id+'_'+content_pos+'_'+block_pos+'_'+block.block_data.data[j][i];
					indexed_desc_list[j]= block.block_data.data[j][i];
				}
			}
		}
		html+='</tr></thead>';
		html+='<tbody>';
		console.time('___render_block_summary()>tablebody');
		for (var j = 0; j <= num_rows-1; j++) {
			html+='<tr>';
			for (var i = 0; i < num_columns; i++) {
				html+=apply_col_render(summary_columns[i].renderer,block.block_data.data[j],i,summary_columns_positions,j);
			}
			html+='</tr>';
		}
		console.timeEnd('___render_block_summary()>tablebody');
		html+='</tbody>';
		html+='</table>';
		html+='</div>';

		// Create datatable object;
		datatable.indexed_list=indexed_list;
		datatable.indexed_desc_list=indexed_desc_list;
		datatable.html_element_id=datatable_id;
		datatable.data=block.block_data.data;
		datatable.columns=block.block_data.columns;
		content_items[item.json_data.item_id].steps[step.step_id].datatables.push(datatable);

		console.timeEnd('___render_block_summary()');
		return html;
	}
	// -----------------------------------------------------------------------------------------------------
	render_block_piechart = function(item,step_pos,content_pos,block_pos) {
		var html =[];
		var step = item.json_data.item_steps[step_pos];
		var element = step.step_contents[content_pos];
		var block = element.content_blocks[block_pos];
		var num_rows=block.block_data.data.length;
		var id_graph='';
		html.push('<div class="block_piechart">');

		var graph_data =[];
		var num_rows=block.block_data.data.length;
		var total_value='';
		for (var i = 0; i < num_rows; i++) {
			if (block.block_data.index[i]=='total') {
				total_value=block.block_data.data[i].toFixed(3);
			}
			var graph_data_el = {};
			if (block.block_data.index[i]!='total' & block.block_data.data[i]!=0) {
				graph_data_el.name = block.block_data.index[i];
				graph_data_el.value = block.block_data.data[i];
				graph_data.push(graph_data_el);
			}
		}
		html.push('<h1 class="block_title">'+get_block_description(block.block_type)+': <span class="block_title_extra">'+total_value+'s</span></h1>');

		if (graph_data.length<2) {
			html.push('</div>');
			return html.join('');
		}


		id_graph='chart_'+step_pos+'_'+content_pos+'_'+block_pos+'_'+Date.now();
		html.push('<div class="echart_chart" id="'+id_graph+'" style="text-aling:center;display: flex;widjth: 230px; height: 128px;"></div>');

		var graph_echart = new Object();
		graph_echart.html_element_id=id_graph;
		graph_echart.options= {
			series: {
				type: 'pie',
				radius: ['35%', '43%'],
				silent: true,
				startAngle: 90,

				color:['#49a5e6','#ea9234', '#f6cb51', '#e65068'],
				label: {
				  verticalAlign: 'middle',
				  align: 'left',
				  fontSize : 11,
				  formatter : function (params){
					return  params.name.replace(' ','\n') + '\n' + params.value.toFixed(2) + 's';
				  }
				},
				labelLine: {show:false,length:0,length2:7},
				options: {
					maintainAspectRatio: true,responsive: true
				},
				data: graph_data
			}

		}
		content_items[item.json_data.item_id].steps[step.step_id].graphs.push(graph_echart);
		html.push('</div>');
		return html.join('');
	}

	// -----------------------------------------------------------------------------------------------------
	render_block_big_number = function(item,step_pos,content_pos,block_pos,big_elements=999) {
		var step = item.json_data.item_steps[step_pos];
		var element = step.step_contents[content_pos];
		var block = element.content_blocks[block_pos];

		var html =[];
		var num_rows=block.block_data.data.length;
		html.push('<div class="block_big_number">');
		html.push('<h1 class="block_title">'+get_block_description(block.block_type)+'</h1>');
		html.push('<div>');

		var big_count=Math.min(num_rows,big_elements);
		for (var i = 0; i < big_count; i++) {
			html.push('<div>');
			html.push('<span class="value">'+block.block_data.data[i].toLocaleString('en-US')+'</span>');
			html.push('<br/>');
			html.push('<span class="prop">'+block.block_data.index[i]+'</span>');
			html.push('</div>');
		}
		html.push('</div>');
		html.push('</div>');

		if (big_count<num_rows) {
			html.push('<div class="block_table" style="margin-top: -15px;">');
			html.push('<table>');
			for (var i = big_count; i < num_rows; i++) {
				html.push('<tr><td>'+block.block_data.index[i]+'</td><td class="value">'+block.block_data.data[i].toLocaleString('en-US')+'</td></tr>');
			}
			html.push('</table>');
			html.push('</div>');
		}


		return html.join('');
	}
	// -----------------------------------------------------------------------------------------------------
	render_block_table = function(item,step_pos,content_pos,block_pos) {
		var step = item.json_data.item_steps[step_pos];
		var element = step.step_contents[content_pos];
		var block = element.content_blocks[block_pos];

		var html =[];
		var num_rows=block.block_data.data.length;
		html.push('<div class="block_table">');
		html.push('<h1 class="block_title">'+get_block_description(block.block_type)+'</h1>');
		html.push('<table>');
		for (var i = 0; i < num_rows; i++) {
			html.push('<tr><td>'+block.block_data.index[i]+'</td><td class="value">'+block.block_data.data[i].toLocaleString('en-US')+'</td></tr>');
		}
		html.push('</table>');
		html.push('</div>');
		return html.join('');
	}
	// -----------------------------------------------------------------------------------------------------
	render_block_raw = function(item,step_pos,content_pos,block_pos) {
		var html =[];
		var step = item.json_data.item_steps[step_pos];
		var element = step.step_contents[content_pos];
		var block = element.content_blocks[block_pos];
		var num_rows=block.block_data.data.length;
		html.push('<div class="block_raw">');
		html.push('<h1 class="block_title">'+get_block_description(block.block_type)+'</h1>');
		for (var i = 0; i < num_rows; i++) {
			html.push(block.block_data.index[i]+': '+block.block_data.data[i]+'<br/>');
		}
		html.push('</div>');
		return html.join('');
	}
	// -----------------------------------------------------------------------------------------------------
	render_block_config = function(item,step_pos,content_pos,block_pos) {
		var html =[];
		var step = item.json_data.item_steps[step_pos];
		var element = step.step_contents[content_pos];
		var block = element.content_blocks[block_pos];
		var num_rows=block.block_data.data.length;
		html.push('<div class="block_config">');
		html.push('<h1 class="block_title">'+get_block_description(block.block_type)+'</h1>');
		for (var i = 0; i < num_rows; i++) {
			html.push('<span class="config_option">'+block.block_data.index[i]+':</span> ');
			html.push('<span class="config_value">'+block.block_data.data[i]+'</span><br/>');
		}
		html.push('</div>');
		return html.join('');
	}


// -----------------------------------------------------------------------------------------------------
// Render element types;
// -----------------------------------------------------------------------------------------------------

	// -----------------------------------------------------------------------------------------------------
	render_element_stats = function(item,step_pos,content_pos) {
		console.time('__render_element_stats()');
		var element_fragment = document.createElement('div');
		var step = item.json_data.item_steps[step_pos];
		var element = step.step_contents[content_pos];
		var html=[];

		var num_blocks=element.content_blocks.length;
		for (var i = 0; i < num_blocks; i++) {
			var block_renderer = assing_block_render(item.json_data.item_type,step.step_type,element.content_type,element.content_blocks[i].block_type);
			html.push(apply_block_render(block_renderer,item,step_pos,content_pos,i));
		}
		element_fragment.innerHTML=html.join('');
		console.timeEnd('__render_element_stats()');
		return element_fragment;
	}

	// -----------------------------------------------------------------------------------------------------
	render_element_summary = function(item,step_pos,content_pos) {
		console.time('__render_element_summary()');
		var element_fragment = document.createElement('div');
		var step = item.json_data.item_steps[step_pos];
		var element = step.step_contents[content_pos];
		var html=[];

		var num_blocks=element.content_blocks.length;
		for (var i = 0; i < num_blocks; i++) {
			var block_renderer = assing_block_render(item.json_data.item_type,step.step_type,element.content_type,element.content_blocks[i].block_type);
			html.push(apply_block_render(block_renderer,item,step_pos,content_pos,i));
		}
		element_fragment.innerHTML=html.join('');
		console.timeEnd('__render_element_summary()');
		return element_fragment;
	}

	render_element_basic = function(item,step_pos,content_pos) {
		console.time('__render_element_basic()');
		var element_fragment = document.createElement('div');
		element_fragment.innerHTML=JSON.stringify(element.data);
		console.timeEnd('__render_element_basic()');
		return element_fragment;
	}




// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

	// -----------------------------------------------------------------------------------------------------
	render_element = function(item,step_pos,content_pos) {
		var element_fragment = document.createElement('div');
		var step = item.json_data.item_steps[step_pos];
		var element = step.step_contents[content_pos];
		element_fragment.setAttribute('class', item.json_data.item_type+' '+step.step_type+' '+element.content_type);


		switch(element.content_type) {
			case 'stats':
				element_fragment.appendChild(render_element_stats(item,step_pos,content_pos));
				break;
			case 'config':
				element_fragment.appendChild(render_element_stats(item,step_pos,content_pos));
				break;
			case 'summary':
				element_fragment.appendChild(render_element_summary(item,step_pos,content_pos));
				break;
			default:
				element_fragment.appendChild(render_element_basic(item,step_pos,content_pos));
		}
		return element_fragment;
	}
	// -----------------------------------------------------------------------------------------------------
	// 2018/06/28
	prepare_step_data = function(item,step_pos) {

		var step_fragment = document.createElement('div');
		step_fragment.setAttribute('class', 'step');

		var step_layout_parts=create_step_layout_parts(item);
		var step_contents=item.json_data.item_steps[step_pos].step_contents;
		// Render elements
		for (var i = 0; i < step_contents.length; i++) {
			var element_fragment = render_element(item,step_pos,i);
			step_layout_parts[step_contents[i].content_position].appendChild(element_fragment);
		}
		step_fragment.appendChild(join_step_layout_parts(item,step_layout_parts));
		return step_fragment;
	}

	// -----------------------------------------------------------------------------------------------------
	// 2018/06/28
	prepare_item_data = function(item_id,step_id) {
		var document_fragment = document.createDocumentFragment();
		// check if it was already clicked and processed
		if (typeof content_items[item_id] == "undefined") {
			item= new Object();
			item.json_data=JSON.parse(json_items_data[item_id]);
			item.steps={};
			item.graphs=[];
			item.datatables=[];
			item.status='parsed';
			json_items_data[item_id]=null;
			content_items[item_id]=item;
		}
		else {
			item=content_items[item_id];
		}

		// step object;
		if (typeof item.steps[step_id] == "undefined") {
			step = new Object();
			step.graphs=[];
			step.datatables=[];
			item.steps[step_id]=step;
		}
		console.log(item.steps[step_id]);


		// search step_id in json_data
		var step_pos=item.json_data.item_steps.findIndex(o => o.step_id === step_id);
		if (step_pos == -1) {
			console.log('GRMlab: ERROR, step_id '+step_id+' not found in json_data');
		}
		document_fragment.appendChild(prepare_step_data(item,step_pos));
document_fragment.innerHTML
		//console.log(item.steps[step_id]);

		return document_fragment;
	}

	// -----------------------------------------------------------------------------------------------------

	// Init particles
	// -----------------------------------------------------------------------------------------------------
	// 2018/06/28
	$(document).ready(function () {
		console.time('Particles');
		window.pJSDom[0].pJS.particles.move.enable = true;
		seed1=1;seed2=1;seed3=1;
		window.pJSDom[0].pJS.fn.particlesRefresh();
		console.timeEnd('Particles');
	});

	// Global
	// -----------------------------------------------------------------------------------------------------
	// 2018/06/28
	var content_items = new Object();

	var elem_main = document.getElementById('main');
	var elem_main_content = document.getElementById('main_content');
	var elem_toc = document.getElementById('toc');
	var elem_toc_icons = document.getElementById('toc_icons');
	var elem_particles = document.getElementById('particles');
	var elem_loading = document.getElementById('loading');

	var step_selected;
	var sidebarSelected;
	var selected_element= document.getElementById('main');
	var selected_id= 'main';

	initial_layout();

	// On sidebar click
	// -----------------------------------------------------------------------------------------------------
	// 2018/06/28
	$('#sidebar').on('click', 'a', function(e) {
		// TODO: Change this!
		$(sidebarSelected).removeClass('active');
		if (typeof sidebarSelected != "undefined")
		{
			if (sidebarSelected.parentElement.getAttribute('class')=='sidebar_subitem') {
				$(sidebarSelected.parentElement.parentElement.previousSibling).removeClass('active');
			}
			else if (sidebarSelected.parentElement.getAttribute('class')=='sidebar_item') {
				$(sidebarSelected.nextSibling.firstChild.firstChild).removeClass('active');
			}
		}


		$(this).addClass('active').siblings();
		if (this.parentElement.getAttribute('class')=='sidebar_item') {
			$(this.nextSibling.firstChild.firstChild).addClass('active');
		}


		sidebarSelected=this;

		if (this.parentElement.getAttribute('class')=='sidebar_subitem') {
			$(this.parentElement.parentElement.previousSibling).addClass('active').siblings();
		}

		clicked_target= this.getAttribute('href').substr(1);
		if (clicked_target==''){ clicked_target='main';}
		if (clicked_target!='main'){
			window.pJSDom[0].pJS.particles.move.enable = false;
			elem_particles.style.display = "none";
			elem_loading.style.display = "block";
		}
		selected_element.style.display = 'none';

		setTimeout(function() {change_item(clicked_target);}, 0);
	});

	// -----------------------------------------------------------------------------------------------------
	// 2018/06/28
	change_item = function(clicked_id) {
		console.time('change_item');
		var clicked_element=document.getElementById(clicked_id);

		if (selected_id!='main'){
			console.time('_remove_element()');
			var selected_item_id=selected_id.split('_')[0];
			var selected_step_id=selected_id.split('_')[1];

			for (var i = 0; i < content_items[selected_item_id].steps[selected_step_id].graphs.length; i++) {
				var graph_element_id=document.getElementById(content_items[selected_item_id].steps[selected_step_id].graphs[i].html_element_id);
				console.log('GRAPH DES:'+graph_element_id);
				// TODO: !!!!!!!! eroor en clicks rapidos;
				window.echarts.getInstanceById(graph_element_id.getAttribute('_echarts_instance_')).dispose();
				//content_items[selected_item_id].steps[selected_step_id].graphs[i].chart_obj.dispose();
			}
			while (selected_element.firstChild) {
				selected_element.removeChild(selected_element.firstChild);
			}
			for (var i = 0; i < content_items[selected_item_id].steps[selected_step_id].datatables.length; i++) {
				content_items[selected_item_id].steps[selected_step_id].datatables[i].table_obj.destroy();
			}

			content_items[selected_item_id].steps[selected_step_id].datatables=[];
			content_items[selected_item_id].steps[selected_step_id].graphs=[];

			console.timeEnd('_remove_element()');
		}

		selected_element=clicked_element;
		selected_id=clicked_id;

		if (selected_id=='main'){
			elem_particles.style.display = "block";
			seed1=1;seed2=1;seed3=1;
			window.pJSDom[0].pJS.particles.move.enable = true;
			window.pJSDom[0].pJS.fn.particlesRefresh();
		}
		else {
			console.time('_prepare_item_data()');
			var clicked_item_id=clicked_id.split('_')[0];
			var clicked_step_id=clicked_id.split('_')[1];

			var new_content = prepare_item_data(clicked_item_id,clicked_step_id);
			console.timeEnd('_prepare_item_data()');

			console.time('_append_content()');
			document.getElementById(clicked_id).appendChild(new_content);
			console.timeEnd('_append_content()');

			console.time('_init_datatable()');
			for (var i = 0; i < content_items[clicked_item_id].steps[clicked_step_id].datatables.length; i++) {
				//console.log(content_items[clicked_item_id].steps[clicked_step_id].datatables[i].html_element_id);
				var table_obj = $('#'+content_items[clicked_item_id].steps[clicked_step_id].datatables[i].html_element_id).DataTable({
					"dom": "fBritS", buttons: [ 'csv' ],
					processing : true,
					scrollY: "100vh" ,scrollX:true,scroller:true,"paging":true,"autoWidth": false,
					"deferRender": true,"searchDelay": 250,"stripeClasses": [ ],"order": []
				});
				content_items[clicked_item_id].steps[clicked_step_id].datatables[i].table_obj=table_obj;
				resizeColumns(table_obj);
				//visibility:hidden;
				//new $.fn.dataTable.FixedColumns( tabla );
				//setTimeout(function() {table_obj.columns.adjust();}, 0);
			}
			console.timeEnd('_init_datatable()');

			console.time('_init_charts()');
			for (var i = 0; i < content_items[clicked_item_id].steps[clicked_step_id].graphs.length; i++) {
				//console.log(content_items[clicked_item_id].steps[clicked_step_id].graphs[i].html_element_id);
				showGraph(content_items[clicked_item_id].steps[clicked_step_id].graphs[i]);
			}
			console.timeEnd('_init_charts()');
		}
		// ???? timeout or new_content
		//setTimeout(function() {
			selected_element.style.display = "block";
			elem_loading.style.display = "none";
		//}, 0);


		console.timeEnd('change_item');
	};

	// resize datatable
	function resizeColumns(datatable) {
		setTimeout(function() {datatable.columns.adjust();}, 0);
	}

		$(document).on('click','.step_list li',function(){
			// TODO: Change this!
			$(step_selected).removeClass('active');
			$(this).addClass('active').siblings();
			step_selected=this;
		});

		$('#sidebarCollapse').on('click', '', function(e) {
			$(this).toggleClass('icon_left');
			$(this).toggleClass('icon_right');
			$('#sidebar, #content').toggleClass('active');
			$('.collapse.in').toggleClass('in');
			$('a[aria-expanded=true]').attr('aria-expanded', 'false');
		});




// ---------------------------------------------------------------------------------
rounding = function(x) {
	var abs_x=Math.abs(x);
	switch (true) {
		case (abs_x == 0): return 0;
		case (abs_x < 0.01): return 4;
		case (abs_x < 0.1): return 3;
		case (abs_x < 1): return 2;
		case (abs_x < 10): return 2;
		default: return 0;
	}
}

format_number = function(x,type='str') {
	if (x == null) return '';
	if (type=='%') return x.toLocaleString('en-US',{  minimumFractionDigits:2, maximumFractionDigits:2 });
	if (type=='str') return x.toLocaleString('en-US',{ minimumFractionDigits:rounding(x), maximumFractionDigits:rounding(x) });
	return x.toFixed(rounding(x));
}

fmt_num = function(x,type='str') {
	if (x == null) return '';
	if (type=='%.1') return x.toLocaleString('en-US',{style: "percent", minimumFractionDigits:0, maximumFractionDigits:1 });
	if (type=='%') return x.toLocaleString('en-US',{style: "percent", minimumFractionDigits:2, maximumFractionDigits:2 });
	if (type=='0') return x.toLocaleString('en-US',{ minimumFractionDigits:0, maximumFractionDigits:0 });
	if (type=='2') return x.toLocaleString('en-US',{ minimumFractionDigits:2, maximumFractionDigits:2 });
	if (type=='4') return x.toLocaleString('en-US',{ minimumFractionDigits:4, maximumFractionDigits:4 });
	if (type=='str') return x.toLocaleString('en-US',{ minimumFractionDigits:rounding(x), maximumFractionDigits:rounding(x) });
	return x.toFixed(rounding(x));
}

// Assesment
// -------------------------------------------------------------------------
/*
assesment = function(module,step, ) {
	switch (true) {
		case (info_stability>0.16 || info_stability_w>0.16): return 'red';
		case (info_stability>0.10 || info_stability_w>0.10): return 'orange';
	}
	return '';
}
*/

asses_stability = function(info_stability,info_stability_w) {
	switch (true) {
		case (info_stability>0.16 || info_stability_w>0.16): return 'red';
		case (info_stability>0.10 || info_stability_w>0.10): return 'orange';
	}
	return '';
}
asses_months_no_info = function(months,informed_months) {
	switch (true) {
		case (informed_months/months < 0.75): return 'red';
		case (months-informed_months > 0): return 'orange';
	}
	return '';
}

asses_univariate = function(variable,values) {
	switch (variable) {
		case 'missing_perc':
			switch (true) {
				case (values[0] > 0.95): return 'red';
				case (values[0] > 0.75): return 'orange';
			}
			return;
		case (months-informed_months > 0): return 'orange';
	}
	return '';
}



// -------------------------------------------------------------------------
univariate_vd_catnom = function(json_data) {

	var results = {
		html:  '',
		graphs: []
	};
	var html =[];
	var v_data=json_data.data;
	var series=[];

	// Content
	// -------------------------------------------------------------------------
	html.push('<div class="c ttg">');

	html.push('<div>');
	html.push('<h1 class="modal_title">'+v_data.name);
	html.push('<span class="italic grey"> ('+v_data.type+')</span>');
	html.push('</h1>');
	html.push('</div>');

	var informed=v_data['informed records'];
	var missing=v_data['missing records'];
	var special=v_data['special records'];
	var total = informed + missing + special;
	var informed_perc=(informed/total);
	var missing_perc=(missing/total);
	var special_perc=(special/total);

	var asses_missing_perc = asses_univariate('missing_perc',[missing_perc]);

	html.push('<div class="block_table_2" style="">');

	html.push('<h1 class="block_subtitle">Information level</h1>');
	html.push('<table style="width:75%;margin:10px 0px 20px 0px;">');
	html.push('<tr><td>Missing:</td><td class="value right '+asses_missing_perc+'">');
	html.push(fmt_num(missing_perc,'%')+'</td>');
	html.push('<td>('+fmt_num(missing,'0')+')</td></tr>');
	html.push('<tr><td>Special:</td><td class="value right">');
	html.push(fmt_num(special_perc,'%')+'</td>');
	html.push('<td>('+fmt_num(special,'0')+')</td>');
	html.push('</tr>');
	html.push('<tr><td>Informed:</td><td class="value right">');
	html.push(fmt_num(informed_perc,'%')+'</td>');
	html.push('<td>('+fmt_num(informed,'0')+')</td>');
	html.push('</tr>');
	html.push('</table>');

	var special_list=v_data['unique special values'];
	var special_catnum=special_list.length;
	var informed_cat=v_data['categories'];
	var HHI=v_data['HHI'];
	var HHI_norm=v_data['HHI (normalized)'];
	html.push('<h1 class="block_subtitle">Distribution</h1>');


	var special_li = '';
	if (special_catnum > 0) {
		special_li += '<div class="tooltip icon_ellipsis-h">';
		special_li += '<span class="tooltiptext left">';
		special_li += '<span class="value_small">Values: </span>';
		special_li += special_list.join(", ")+'</span></div>';
	}

	html.push('<table class="force_nowrap" style="width:75%;margin:10px 0px 20px 0px;">');
	html.push('<tr><td>Specials:</td>');
	html.push('<td class="right"><span class="value">'+fmt_num(special_catnum,'0')+'</span>');
	html.push(' distinct values '+special_li+'</td><td></td></tr>');
	html.push('<tr><td>Informed:</td>');
	html.push('<td class="right"><span class="value">'+fmt_num(informed_cat,'0')+'</span>');
	html.push(' distinct values</td><td></td></tr>');
	html.push('</table>');


	html.push('<table style="width:75%;margin:10px 0px 20px 0px;">');
	html.push('<tr><td>HHI:</td><td class="value right">');
	html.push(fmt_num(HHI)+'</td>');
	html.push('<td>HHI (normalized):</td><td class="value right">');
	html.push(fmt_num(HHI_norm)+'</td>');
	html.push('</tr>');
	html.push('</table>');

	html.push('</div>');

	// Chart distribution
	// -------------------------------------------------------------------------
	var graph = new Object();
	graph.html_element_id='chart_'+Math.random().toString(36).substr(2, 12);
	var hist_info_col=v_data['hist_info_col'].slice().reverse();
	var hist_info_pos=v_data['hist_info_pos'].slice().reverse();
	var top10_count=v_data['hist_info_col'].slice().reduce((a, b) => a + b, 0);
	var others_count=informed-top10_count;

	var serie_data= []
	for (var i = 0; i < hist_info_col.length; i++) {
		//serie_data.push({value:hist_info_col[i],itemStyle:{color:'#4667a9'}});
		serie_data.push({value:hist_info_col[i],itemStyle:{color:'#69b0e4'}});
	}

	if (others_count>0) {
		serie_data.unshift({value:others_count,itemStyle:{color:'#a9a9a9'}});
		hist_info_pos.unshift('Others');
	}

	graph.options = {
		tooltip: {
			axisPointer: {show:false},
			trigger: 'axis',
			zzformatter: "{a} <br/>{b} : {c} ({d}%)",
			aaformatter: function(params){
				console.log(params);
				return "{a}";
		}
		},
		legend: {},
		grid: {left:'40',right:'15',bottom:'20',top:'25',},
		yAxis: {
			type: 'category',
			axisLabel: {interval: 0,color:'#8999b9',fontSize: 10},
			axisLine: {lineStyle: {color:'#8999b9'}},
			data: hist_info_pos
		},
		xAxis: {
			splitLine: {show: true,lineStyle: {color: '#455678'}},
			type: 'value',
			axisLabel: {interval: 0,color:'#8999b9',fontSize: 10},
			axisLine: {lineStyle: {color:'#8999b9'}}
		},

		series: [{name:'',type:'bar',data: serie_data,barMaxWidth: 30}]
	};
	results.graphs.push(graph);
	// Chart #a9a9a9
	html.push('<div style="padding-bottom:35px;padding-top:20px;" class="echart_chart" id="'+graph.html_element_id+'"></div>');

	// End c div
	html.push('</div>');

	// Chart Information temporal
	// -------------------------------------------------------------------------
	var graph = new Object();
	graph.html_element_id='chart_'+Math.random().toString(36).substr(2, 12);
	graph.options = {
		tooltip: {
			trigger: 'axis',
			aformatter: "{a} <br/>{b} : {c} ({d}%)",
		},
		legend: {
			symbolKeepAspect:false,itemHeight:4,itemWidth:12,
			icon:'roundRect', bottom:'0',
			data:[
				{name:'informed',icon:'roundRect',textStyle:{fontSize: 10,color:'#49a5e6'}},
				{name:'missing' ,icon:'roundRect',textStyle:{fontSize: 10,color:'#e65068'}},
				{name:'special' ,icon:'roundRect',textStyle:{fontSize: 10,color:'#ea9234'}}
			],
		},
		grid: {left:  '40',right: '15',bottom:'40',top:   '10',},
		xAxis: {
			type: 'category',
			axisLabel: {color:'#8999b9',fontSize: 10},
			axisLine: {
				symbol: ['none', 'arrow'],
				symbolOffset:[0, 8],
				symbolSize:[6, 10],
				lineStyle: {color:'#8999b9'}
			},
			boundaryGap: false,
			data: v_data['dates']
		},
		yAxis: {
			splitLine: {show: true,lineStyle: {color: '#455678'}},
			type: 'value',
			axisLabel: {
				fontSize: 10,
				fontFamily: 'open sans',
				color:'#8999b9',
				formatter: function(value){
					return (value*1)+'%';
				}
			},
			axisLine: {lineStyle: {color:'#8999b9'}}
		},
		color:['#49a5e6','#e65068','#ea9234', '#e65068','#f6cb51'],
		animationDuration: 300,
		series: [
			{
				type:'line',name:'informed',symbol:'none',
				data: v_data['temp_info'].map(x => (x*100).toFixed(2)),
			},
			{
				type:'line',name:'missing',symbol:'none',
				data: v_data['temp_missing'].map(x => (x*100).toFixed(2)),
			},
			{
				type:'line',name:'special',symbol:'none',
				data: v_data['temp_special'].map(x => (x*100).toFixed(2)),
			}
		]
	};
	results.graphs.push(graph);

	// HTML
	html.push('<div class="g1 ttg">');

	html.push('<div><h1 class="block_subtitle">Temporal analysis</h1></div>');

	// Stability
	var info_stability=v_data['information stability (evenly)'];
	var info_stability_w=v_data['information stability (weighted)'];
	var months=v_data['dates'].length;
	var months_no_info=v_data['months w/o information'];
	var informed_months=months - months_no_info;
	var assesment = asses_months_no_info(months,informed_months);

	html.push('<div class="block_table_2">');
	html.push('<table style="width:88%;">');
	html.push('<tr><td>Informed months: ');
	html.push('<span class="value '+assesment+'">'+fmt_num(informed_months));
	html.push(' / '+fmt_num(months)+'</span>');
	html.push('</td>');
	html.push('<td>Information stability:</td>');
	html.push('<td>');
	var assesment = asses_stability(info_stability,info_stability_w);
	html.push('<span class="value '+assesment+'">'+fmt_num(info_stability,'4')+'</span>');
	html.push('<span class="value_small"> (global)</span></td>');
	html.push('<td>');
	html.push('<span class="value '+assesment+'">'+fmt_num(info_stability_w,'4')+'</span>');
	html.push('<span class="value_small"> (weighted)</span></td>');
	html.push('</tr>');
	html.push('</table>');
	html.push('</div>');

	// Chart
	html.push('<div class="echart_chart" id="'+graph.html_element_id+'"></div>');

	html.push('</div>');

	// Chart Values temporal (Categories)
	// -------------------------------------------------------------------------
	graph = new Object();
	graph.html_element_id='chart_'+Math.random().toString(36).substr(2, 12);
	var series = [];
	var data_list = ['temp_c0','temp_c1','temp_c2','temp_c3','temp_c4',
					 'temp_c5','temp_c6','temp_c7','temp_c8','temp_c9'];
	for (var element in data_list) {
		var data_series = v_data[data_list[element]];
		if (data_series.length != 0) {
			series.push(
				{
					name: '"'+v_data.top_cat[element]+'"',
					data: data_series.map(x => fmt_num(x*100)),
					symbol:'none',type:'line',hoverAnimation:false
				}
			);
		}
	}
	if (! v_data['temp_rest'].every(x => x == 0)) {
		series.push(
			{
				name: 'Others',
				data: v_data['temp_rest'].map(x => fmt_num(x*100)),
				symbol:'none',type:'line',hoverAnimation:false, color: '#a9a9a9'
			}
		);
	}
	console.log(series);
	graph.options = {
		tooltip: {
			axisPointer: {show:false},
			trigger: 'axis',
			aformatter: "{a} <br/>{b} : {c} ({d}%)",
			aaformatter: function(params){
				console.log(params);
				return "{a}";
			}
		},
		legend: {
			symbolKeepAspect:false,itemHeight:4,itemWidth:12,
			inactiveColor:'#57698c',textStyle: {color: '#8999b9',fontSize: 10},
			icon:'roundRect', bottom:'0',itemGap:6
		},
		grid: {left:  '40',right: '15',bottom:'57',top:   '10',},
		xAxis: {
			type: 'category',
			axisLabel: {color:'#8999b9',fontSize: 10},
			axisLine: {
				symbol: ['none', 'arrow'],
				symbolOffset:[0, 8],
				symbolSize:[6, 10],
				lineStyle: {color:'#8999b9'}
			},
			boundaryGap: false,
			data: v_data['dates']
		},
		yAxis: {
			type: 'value',
			scale:true,
			axisLabel: {
				color:'#8999b9',
				fontSize: 10,
				formatter: function(value){
					return (value*1)+'%';
				}
			},
			splitLine: {show: true,lineStyle: {color: '#455678'}},
			axisLine: {lineStyle: {color:'#8999b9'}}
		},
		color:[
			'#0074D9','#7FDBFF','#3D9970','#2ECC40','#01FF70','#FFDC00','#FF851B',
			//'#FF4136',
			'#85144b','#F012BE','#B10DC9','#111111','#AAAAAA','#DDDDDD'
		],
		animationDuration: 300,series: series,
	}
	results.graphs.push(graph);

	// HTML
	html.push('<div class="g2 tg">');

	// Stability
	var data_stability=v_data['data values stability (evenly)'];
	var data_stability_w=v_data['data values stability (weighted)'];
	var assesment = asses_stability(data_stability,data_stability_w);

	html.push('<div class="block_table_2">');
	html.push('<table style="width:88%;">');
	html.push('<tr><td>Informed records: ');
	html.push('<span class="value">'+fmt_num(informed)+'</span>');
	html.push('</td>');
	html.push('<td>Values stability:</td>');
	html.push('<td>');
	html.push('<span class="value '+assesment+'">'+fmt_num(data_stability,'4')+'</span>');
	html.push('<span class="value_small"> (global)</span></td>');
	html.push('<td>');
	html.push('<span class="value '+assesment+'">'+fmt_num(data_stability_w,'4')+'</span>');
	html.push('<span class="value_small"> (weighted)</span></td>');
	html.push('</tr>');
	html.push('</table>');
	html.push('</div>');

	// Chart
	html.push('<div class="echart_chart" id="'+graph.html_element_id+'"></div>');

	html.push('</div>');

	results.html=html.join('');
	return results;
}

// -------------------------------------------------------------------------
univariate_vd_num = function(json_data) {

	var results = {
		html:  '',
		graphs: []
	};
	var html =[];
	var v_data=json_data.data;
	var series=[];

	// Content
	// -------------------------------------------------------------------------
	html.push('<div class="c ttg">');

	html.push('<div>');
	html.push('<h1 class="modal_title">'+v_data.name);
	html.push('<span class="italic grey"> ('+v_data.type+')</span>');
	html.push('</h1>');
	html.push('</div>');

	var informed=v_data['informed records'];
	var missing=v_data['missing records'];
	var special=v_data['special records'];
	var total = informed + missing + special;
	var informed_perc=(informed/total);
	var missing_perc=(missing/total);
	var special_perc=(special/total);
	var special_cat=v_data['unique special values'].length;
	var informed_cat=v_data['categories'];
	var asses_missing_perc = asses_univariate('missing_perc',[missing_perc]);

	html.push('<div class="block_table_2" style="">');
	html.push('<h1 class="block_subtitle">Information level</h1>');
	html.push('<table style="width:75%;margin:10px 0px 20px 0px;">');
	html.push('<tr><td>Missing:</td><td class="value right '+asses_missing_perc+'">');
	html.push(fmt_num(missing_perc,'%')+'</td>');
	html.push('<td>('+fmt_num(missing,'0')+')</td></tr>');
	html.push('<tr><td>Special:</td><td class="value right">');
	html.push(fmt_num(special_perc,'%')+'</td>');
	html.push('<td>('+fmt_num(special,'0')+')</td>');
	html.push('</tr>');
	html.push('<tr><td>Informed:</td><td class="value right">');
	html.push(fmt_num(informed_perc,'%')+'</td>');
	html.push('<td>('+fmt_num(informed,'0')+')</td>');
	html.push('</tr>');
	html.push('</table>');

		var min=v_data['min'];
		var mean=v_data['mean'];
		var mode=v_data['mode'];
		var max=v_data['max'];

		var percentile_1=v_data['percentile 1%'];
		var percentile_25=v_data['percentile 25%'];
		var median=v_data['median'];
		var percentile_75=v_data['percentile 75%'];
		var percentile_99=v_data['percentile 99%'];

		var positive=v_data['positive'];
		var negative=v_data['negative'];
		var zeros=v_data['zeros'];
		var std=v_data['std'];

		html.push('<h1 class="block_subtitle">Distribution</h1>');

		html.push('<div style="width:90%">');
		html.push('<table style="margin:10px 0px 20px 0px;">');
		html.push('<tr><td>Negative:</td><td class="value">'+format_number(negative)+'</td>');
		html.push('<td>Zeros:</td><td class="value">'+format_number(zeros)+'</td>');
		html.push('<td>Positive:</td><td class="value">'+format_number(positive)+'</td></tr>');
		html.push('</table>');

		html.push('<table style="margin:10px 0px 20px 0px;float:left;">');
		html.push('<tr><td>Mean:</td><td class="value">'+format_number(mean)+'</td></tr>');
		html.push('<tr><td>Mode:</td><td class="value">'+format_number(mode)+'</td></tr>');
		html.push('<tr><td>P50:</td><td class="value">'+format_number(median)+'</td></tr>');
		html.push('<tr><td>Std:</td><td class="value">'+format_number(std)+'</td></tr>');
		html.push('</table>');

		html.push('<table style="margin:10px 0px 20px 0px;">');
		html.push('<tr><td>Min:</td><td class="value">'+format_number(min)+'</td></tr>');
		html.push('<tr><td>P1:</td><td class="value">'+format_number(percentile_1)+'</td></tr>');
		html.push('<tr><td>P25:</td><td class="value">'+format_number(percentile_25)+'</td></tr>');
		html.push('<tr><td>P75:</td><td class="value">'+format_number(percentile_75)+'</td></tr>');
		html.push('<tr><td>P99:</td><td class="value">'+format_number(percentile_99)+'</td></tr>');
		html.push('<tr><td>Max:</td><td class="value">'+format_number(max)+'</td></tr>');
		html.push('</table>');
		html.push('</div>');

	html.push('</div>');

	var hist_info_col=v_data['hist_info_col'];
	var hist_info_pos=v_data['hist_info_pos'];
	var outliers_high=v_data['outliers high'];
	var outliers_low=v_data['outliers low'];
	var outliers_high_threshold=v_data['outlier high threshold'];
	var outliers_low_threshold=v_data['outlier low threshold'];


	var serie_label= [];
	var serie_data= [];
	for (var i = 0; i < hist_info_col.length; i++) {
		if (hist_info_col[i]>0) {
			serie_data.push({value:hist_info_col[i],itemStyle:{color:'#69b0e4'}});
			serie_label.push({value:fmt_num(hist_info_pos[i]),itemStyle:{color:'#69b0e4'}});
		}
	}

	if (outliers_low>0 | true) {
		serie_data.unshift({value:0,itemStyle:{color:'#a9a9a9'}});
		serie_label.unshift({value:'outlier threshold',itemStyle:{color:'#a9a9a9'}});
		serie_data.unshift({value:outliers_low,itemStyle:{color:'#a9a9a9'}});
		serie_label.unshift({value:fmt_num(outliers_low_threshold),itemStyle:{color:'#a9a9a9'}});
	}
	if (outliers_high>0 | true) {
		serie_data.push({value:0,itemStyle:{color:'#a9a9a9'}});
		serie_label.push({value:'outlier threshold',itemStyle:{color:'#a9a9a9'}});
		serie_data.push({value:outliers_high,itemStyle:{color:'#a9a9a9'}});
		serie_label.push({value:fmt_num(outliers_high_threshold),itemStyle:{color:'#a9a9a9'}});
	}

		var graph = new Object();
		graph.html_element_id='chart_'+Math.random().toString(36).substr(2, 12);
		graph.options = {
			tooltip: {
				axisPointer: {show:false},
				trigger: 'axis',
				zzformatter: "{a} <br/>{b} : {c} ({d}%)",
				aaformatter: function(params){
					console.log(params);
					return "{a}";
				}
			},
			legend: {

			},
			grid: {
				left: '3%',
				right: '4%',
				bottom: '8%',
				top:5,
				containLabel: true
			},
			yAxis: {
				splitLine: {show: true,lineStyle: {color: '#455678'}},
				axisLabel: {
					interval: 0,
					color:'#8999b9',
					fontSize: 10
				},
				axisLine: {show:false,lineStyle: {color:'#8999b9'}},
				name:'# of records',
				type: 'value',
			},
			xAxis: {
				//name:'value',
				type: 'category',
				data: serie_label,
				axisLabel: {
					color:'#8999b9',
					fontSize: 10,
				},
				axisLine: {lineStyle: {color:'#8999b9'}},
			},
			color:['#5abafe','#49a5e6','#3a80b2', '#e65068','#f6cb51'],
			series: [
				{
					name:'',type:'bar',
					data: serie_data,barMaxWidth: 30,
            	markLine: {
            		silent:true,animation:false,
            		label: {position:'middle'},
            		lineStyle: {color:'red',type:'solid'},
            		xindexValue: 0,
					data: [
				    	[
					        {
				            	name: '',symbol:'none',
				            	xAxis: 1,
				            	yAxis: 'min'
				        	},
				        	{	symbol:'none',
					            xAxis: 1,
				            	y: '2%'
				        	}
				    	],
				    	[
					        {
				            	name: '',symbol:'none',
				            	xAxis: serie_label.length-2,
				            	yAxis: 'min'
				        	},
				        	{	symbol:'none',
					            xAxis: serie_label.length-2,
				            	y: '2%'
				        	}
				    	]
					]
            	}
				},
			]
		};
		results.graphs.push(graph);
		html.push('<div class="echart_chart" id="'+graph.html_element_id+'"></div>');

	// end c contents
	html.push('</div>');

	// -------------------------------------------------------------------------
	// Chart Information temporal
	// -------------------------------------------------------------------------
	var graph = new Object();
	graph.html_element_id='chart_'+Math.random().toString(36).substr(2, 12);
	graph.options = {
		tooltip: {
			trigger: 'axis',
			aformatter: "{a} <br/>{b} : {c} ({d}%)",
		},
		legend: {
			symbolKeepAspect:false,itemHeight:4,itemWidth:12,
			icon:'roundRect', bottom:'0',
			data:[
				{name:'informed',icon:'roundRect',textStyle:{fontSize: 10,color:'#49a5e6'}},
				{name:'missing' ,icon:'roundRect',textStyle:{fontSize: 10,color:'#e65068'}},
				{name:'special' ,icon:'roundRect',textStyle:{fontSize: 10,color:'#ea9234'}}
			],
		},
		grid: {left:  '40',right: '15',bottom:'40',top:   '10',},
		xAxis: {
			type: 'category',
			axisLabel: {color:'#8999b9',fontSize: 10},
			axisLine: {
				symbol: ['none', 'arrow'],
				symbolOffset:[0, 8],
				symbolSize:[6, 10],
				lineStyle: {color:'#8999b9'}
			},
			boundaryGap: false,
			data: v_data['dates']
		},
		yAxis: {
			splitLine: {show: true,lineStyle: {color: '#455678'}},
			type: 'value',
			axisLabel: {
				fontSize: 10,
				fontFamily: 'open sans',
				color:'#8999b9',
				formatter: function(value){
					return (value*1)+'%';
				}
			},
			axisLine: {lineStyle: {color:'#8999b9'}}
		},
		color:['#49a5e6','#e65068','#ea9234', '#e65068','#f6cb51'],
		animationDuration: 300,
		series: [
			{
				type:'line',name:'informed',symbol:'none',
				data: v_data['temp_info'].map(x => (x*100).toFixed(2)),
			},
			{
				type:'line',name:'missing',symbol:'none',
				data: v_data['temp_missing'].map(x => (x*100).toFixed(2)),
			},
			{
				type:'line',name:'special',symbol:'none',
				data: v_data['temp_special'].map(x => (x*100).toFixed(2)),
			}
		]
	};
	results.graphs.push(graph);

	// HTML
	html.push('<div class="g1 ttg">');
	html.push('<div><h1 class="block_subtitle">Temporal analysis</h1></div>');

	// Stability
	var info_stability=v_data['information stability (evenly)'];
	var info_stability_w=v_data['information stability (weighted)'];
	var assesment = asses_stability(info_stability,info_stability_w);
	var months=v_data['dates'].length;
	var months_no_info=v_data['months w/o information'];
	var informed_months=months - months_no_info;
	var assesment = asses_months_no_info(months,informed_months);

	html.push('<div class="block_table_2">');
	html.push('<table style="width:88%;">');
	html.push('<tr><td>Informed months: ');
	html.push('<span class="value '+assesment+'">'+fmt_num(informed_months));
	html.push(' / '+fmt_num(months)+'</span>');
	html.push('</td>');
	html.push('<td>Information stability:</td>');
	html.push('<td>');
	html.push('<span class="value '+assesment+'">'+fmt_num(info_stability,'4')+'</span>');
	html.push('<span class="value_small"> (global)</span></td>');
	html.push('<td>');
	html.push('<span class="value '+assesment+'">'+fmt_num(info_stability_w,'4')+'</span>');
	html.push('<span class="value_small"> (weighted)</span></td>');
	html.push('</tr>');
	html.push('</table>');
	html.push('</div>');

	// Chart
	html.push('<div class="echart_chart" id="'+graph.html_element_id+'"></div>');

	html.push('</div>');

	// Chart Values temporal (Categories)
	// -------------------------------------------------------------------------
	graph = new Object();
	graph.html_element_id='chart_'+Math.random().toString(36).substr(2, 12);
	var series = [];

		graph.options = {
			tooltip: {
				axisPointer: {show:false},
				trigger: 'axis',
				aformatter: "{a} <br/>{b} : {c} ({d}%)",
				aaformatter: function(params){
					console.log(params);
					return "{a}";
				}
			},
			legend: {
				symbolKeepAspect:false,itemHeight:4,itemWidth:12,
				inactiveColor:'#999',textStyle: {color: '#ddd',fontSize: 10},
				icon:'roundRect', bottom:'0',itemGap:6,
				data:[
					{name:'p25',icon:'roundRect',textStyle: {color: '#3a80b2'}},
					{name:'mean' ,icon:'roundRect',textStyle: {color: '#49a5e6'}},
					{name:'p75' ,icon:'roundRect',textStyle: {color: '#3a80b2'}}],
			},
			grid: {left:  '40',right: '15',bottom:'57',top:   '10',},
			xAxis: {
				type: 'category',
				axisLabel: {color:'#8999b9',fontSize: 10},
				axisLine: {
					symbol: ['none', 'arrow'],
					symbolOffset:[0, 8],
					symbolSize:[6, 10],
					lineStyle: {color:'#8999b9'}
				},
				boundaryGap: false,
				data: v_data['dates']
			},
			yAxis: {
				type: 'value',
				scale:true,
				axisLabel: {color:'#8999b9',fontSize: 10},
				splitLine: {
					show: true,
					lineStyle: {
						color: '#556688'
					}
				},
				axisLine: {
					lineStyle: {color:'#8999b9'}
				}
			},
			color:['#3a80b2','#49a5e6','#3a80b2', '#3a80b2','#00ccff'],
			animationDuration: 300,
			series: [
				{
					name:'p25',symbol:'none',
					lineStyle: {type:'dotted'},
					type:'line',
					hoverAnimation:false,
					data: v_data['temp_percentile25'].map(x => format_number(x,'num'))
				},
				{
					name:'mean',symbol:'none',
					type:'line',
					hoverAnimation:false,
					data: v_data['temp_median'].map(x => format_number(x,'num'))
				},
				{
					name:'p75',symbol:'none',
					lineStyle: {type:'dotted'},
					type:'line',
					hoverAnimation:false,
					data: v_data['temp_percentile75'].map(x => format_number(x,'num')),
				}
			]
		};

	results.graphs.push(graph);

	// HTML
	html.push('<div class="g2 tg">');

	// Stability
	var data_stability=v_data['data values stability (evenly)'];
	var data_stability_w=v_data['data values stability (weighted)'];
	var assesment = asses_stability(data_stability,data_stability_w);

	html.push('<div class="block_table_2">');
	html.push('<table style="width:88%;">');
	html.push('<tr><td>Informed records: ');
	html.push('<span class="value">'+fmt_num(informed)+'</span>');
	html.push('</td>');
	html.push('<td>Values stability:</td>');
	html.push('<td>');
	html.push('<span class="value '+assesment+'">'+fmt_num(data_stability,'4')+'</span>');
	html.push('<span class="value_small"> (global)</span></td>');
	html.push('<td>');
	html.push('<span class="value '+assesment+'">'+fmt_num(data_stability_w,'4')+'</span>');
	html.push('<span class="value_small"> (weighted)</span></td>');
	html.push('</tr>');
	html.push('</table>');
	html.push('</div>');

	// Chart
	html.push('<div class="echart_chart" id="'+graph.html_element_id+'"></div>');

	html.push('</div>');
	// -------------------------------------------------------------------------

	results.html=html.join('');
	return results;
}

//-----------------------------------------------------------------------------
// eCharts
//-----------------------------------------------------------------------------

// Initialize and resizze chart;
function showGraph(graph,timeout=1) {
	setTimeout(function() {
		var graph_element_id=document.getElementById(graph.html_element_id);
		var myChart = echarts.init(graph_element_id, null,{renderer: 'svg'}).setOption(graph.options);
	}
	, timeout);
}
function showGraphNow(graph) {
	var graph_element_id=document.getElementById(graph.html_element_id);
	var myChart = echarts.init(graph_element_id, null,{renderer: 'svg'}).setOption(graph.options);
}

function refreshGraphNow(graph) {
	var graph_element_id=document.getElementById(graph.html_element_id);
	graph_element_id.firstElementChild.style.width='100%';
	window.echarts.getInstanceById(graph_element_id.getAttribute('_echarts_instance_')).resize();
	graph_element_id.firstElementChild.style.width='100%';
}

// Dispose chart;
function disposeGraph(graph) {
	var graph_element_id=document.getElementById(graph.html_element_id);
	window.echarts.getInstanceById(graph_element_id.getAttribute('_echarts_instance_')).dispose();
}

//-----------------------------------------------------------------------------

var keyboard_disabled=false;
var elem_modal_div = document.getElementById('modal_div');

//-----------------------------------------------------------------------------
// Sidebar
//-----------------------------------------------------------------------------

// User interaction;
//-----------------------------------------------------------------------------

// Collapse sidebar;
$(document).keyup(function(e) {
	if (keyboard_disabled==true) return;
	var is_modal_active = elem_modal_div.style.display;
	if (is_modal_active !='none' & is_modal_active !='') return;
	keyboard_disabled=true;
	// Collapse sidebar;
	if ( e.originalEvent.altKey==true & e.originalEvent.keyCode=="186") {
		$('#sidebarCollapse').toggleClass('icon_left');
		$('#sidebarCollapse').toggleClass('icon_right');
		$('#sidebar, #content').toggleClass('active');
		$('.collapse.in').toggleClass('in');
		$('a[aria-expanded=true]').attr('aria-expanded', 'false');
	}
	setTimeout(function() {keyboard_disabled=false;}, 0);
});

//-----------------------------------------------------------------------------
// Modal boxes
//-----------------------------------------------------------------------------


// User interaction;
//-----------------------------------------------------------------------------

$(document).on('click','.modal_box_item',function(){
	modalObject.change_item(this.getAttribute('pos'));
});

// Open modal box;
$(document).on('click','.modal_button',function(){
	var content_id=this.getAttribute('id');
	var item_id = content_id.split('_')[0];
	var step_id = content_id.split('_')[1];
	var datatable_pos = content_id.split('_')[3];
	var datatable = content_items[item_id].steps[step_id].datatables[datatable_pos];
	var table_obj = datatable.table_obj;
	var filtered_rows = table_obj.rows({order:'applied', search:'applied'})[0];
	var indexed_list = datatable.indexed_list;
	var indexed_desc_list = datatable.indexed_desc_list;

	var nav_list = [];
	var nav_desc_list = [];
	for (var i = 0; i < filtered_rows.length; i++) {
		nav_list.push(indexed_list[filtered_rows[i]]);
		nav_desc_list.push(indexed_desc_list[filtered_rows[i]]);
	}
	var nav_pos = nav_list.indexOf(content_id);

	modalObject = new modalBox(elem_modal_div,nav_list,nav_desc_list,nav_pos);
	modalObject.show_modal();
	console.log(modalObject);
});

// Close modal box;
$(document).on('click','#modal_close',function(){
	modalObject.close();
});

window.onclick = function(event) {
	if (event.target == elem_modal_div) {
		modalObject.close();
	}
}

$(document).keyup(function(e) {
	var is_modal_active = elem_modal_div.style.display;
	if (e.originalEvent.code === "Escape" & is_modal_active !='none') {
		modalObject.close();
		setTimeout(function() {keyboard_disabled=false;}, 30);
	}
});

// Modal box navigation;
$(document).keyup(function(e) {
	if (keyboard_disabled==true) return;
	if (elem_modal_div.style.display =='none') return;
	if (elem_modal_div.style.display =='') return;
	keyboard_disabled=true;
	// Left: Show previous;
	if ( e.originalEvent.keyCode=="37" | e.originalEvent.keyCode=="38") {
		modalObject.change_item(Math.max(0,modalObject.nav_pos-1));
	}
	// Right: Show next;
	else if ( e.originalEvent.keyCode=="39" | e.originalEvent.keyCode=="40") {
		modalObject.change_item(Math.min(modalObject.nav_list.length-1,modalObject.nav_pos+1));
	}
	// Home
	else if ( e.originalEvent.keyCode=="36") {
		modalObject.change_item(0);
	}
	// End
	else if ( e.originalEvent.keyCode=="35") {
		modalObject.change_item(modalObject.nav_list.length-1);
	}
	// Pageup
	else if ( e.originalEvent.keyCode=="33") {
		modalObject.change_item(Math.max(0,modalObject.nav_pos-14));
	}
	// Pagedown
	else if ( e.originalEvent.keyCode=="34") {
		modalObject.change_item(Math.min(modalObject.nav_list.length-1,modalObject.nav_pos+14));
	}
	setTimeout(function() {keyboard_disabled=false;}, 0);
});

window.onresize = function() {
	if (typeof modalObject != "undefined") {
		modalObject.refresh_content_assets()
	}
}

// Modal box class definition
//-----------------------------------------------------------------------------
class modalBox {

  //---------------------------------------------------------------------------
  constructor(elem_container,nav_list,nav_desc_list,nav_pos) {
	this.elem_container = elem_container;
	this.elem_nav_panel = elem_container.firstElementChild.lastElementChild.firstElementChild;
	this.elem_content = elem_container.firstElementChild.lastElementChild.lastElementChild;
	this.nav_list = nav_list;
	this.nav_desc_list = nav_desc_list;
	this.nav_pos = parseInt(nav_pos);
	this.json_data= '';
	this.content_html = '';
	this.nav_html = '';
	this.graphs = [];

	this.set_json_data();
	this.set_content_html();
	this.set_nav_html();
  }

  //---------------------------------------------------------------------------
  set_json_data() {
	var selected_id = this.nav_list[this.nav_pos];
	if (typeof content_items[selected_id] == "undefined") {
		item= new Object();
		var json = json_items_data[selected_id];
		if (typeof json != "undefined") {
			item.json_data=JSON.parse(json.replace(/\bNaN\b/g, "null"));
			item.status='parsed';
		}
		else { item.json_data=''}
		item.graphs=[];
		content_items[selected_id]=item;
	}
	this.json_data = content_items[selected_id].json_data;
  }

  //---------------------------------------------------------------------------
  set_content_html() {
  	if (this.json_data == '') {return;}
	var selected_id = this.nav_list[this.nav_pos];
	var var_type = this.json_data.data.type;
	var html=[];
	if (var_type =='ordinal' | var_type =='numerical') {
		//html.push(render_detailed_block_overview(selected_id));
		//html.push(render_detailed_block_distribution(selected_id));
		var contents = univariate_vd_num(this.json_data);
		html.push(contents.html);
		for (var i = 0; i < contents.graphs.length; i++) {
			this.graphs.push(contents.graphs[i]);
		}
	}
	else if (var_type =='nominal' | var_type =='categorical') {
		var contents = univariate_vd_catnom(this.json_data);
		html.push(contents.html);
		for (var i = 0; i < contents.graphs.length; i++) {
			this.graphs.push(contents.graphs[i]);
		}
	}
	this.content_html = html.join('');
  }

  //---------------------------------------------------------------------------
  set_nav_html() {
	var html=[];
	html.push('<ul>');
	for (var i = 0; i < this.nav_desc_list.length; i++) {
		html.push('<li class="modal_box_item" pos="'+i+'">');
		html.push(this.nav_desc_list[i]);
		html.push('</li>');
	}
	html.push('</ul>');
	this.nav_html = html.join('');
  }

  //---------------------------------------------------------------------------
  attach_content_html() {
	this.elem_content.innerHTML=this.content_html;
  }

  //---------------------------------------------------------------------------
  show_content_assets() {
	for (var i = 0; i < this.graphs.length; i++) {
		showGraph(this.graphs[i],0);
	}
  }

  //---------------------------------------------------------------------------
  refresh_content_assets() {
	for (var i = 0; i < this.graphs.length; i++) {
		refreshGraphNow(this.graphs[i]);
	}
  }

  //---------------------------------------------------------------------------
  dettach_content_html() {
	this.elem_content.innerHTML='';
  }

  //---------------------------------------------------------------------------
  destroy_content_assets() {
	for (var i = 0; i < this.graphs.length; i++) {
		disposeGraph(this.graphs[i]);
	}
	this.graphs=[];
  }

  //---------------------------------------------------------------------------
  attach_nav_html() {
	this.elem_nav_panel.innerHTML=this.nav_html;
  }

  //---------------------------------------------------------------------------
  dettach_nav_html() {
	this.elem_nav_panel.innerHTML='';
  }

  //---------------------------------------------------------------------------
  change_item(item_pos) {
  	if (this.nav_pos==item_pos) return;
	this.destroy_content_assets();
	this.dettach_content_html();
	this.elem_nav_panel.firstChild.childNodes[this.nav_pos].classList.remove("active");
	this.elem_nav_panel.firstChild.childNodes[item_pos].classList.add("active");
	this.nav_pos=parseInt(item_pos);
	this.set_json_data();
	this.set_content_html();
	this.attach_content_html();
	this.show_content_assets();
	var parent_height = this.elem_nav_panel.firstChild.childNodes[item_pos].offsetParent.offsetHeight;
	var offsetTop = this.elem_nav_panel.firstChild.childNodes[item_pos].offsetTop;
	var offsetHeight = this.elem_nav_panel.firstChild.childNodes[item_pos].offsetHeight;
	if ((offsetTop-this.elem_nav_panel.scrollTop) < (0.12*parent_height)-offsetHeight) {
		this.elem_nav_panel.scrollTop = offsetTop-(0.12*parent_height)+offsetHeight;

	}
	else if ((offsetTop-this.elem_nav_panel.scrollTop) > (0.88*parent_height)) {
		this.elem_nav_panel.scrollTop = offsetTop-(0.88*parent_height);
	}
  }

  //---------------------------------------------------------------------------
  show_modal() {
	this.attach_nav_html();
	this.attach_content_html();
	this.elem_nav_panel.firstChild.childNodes[this.nav_pos].classList.add("active");
	this.elem_container.style.display = "block";
	this.show_content_assets();
	this.elem_container.focus();
  }

  //---------------------------------------------------------------------------
  close() {
	this.elem_container.style.display = "none";
	this.destroy_content_assets();
	this.dettach_nav_html();
	this.dettach_content_html();
  }

}
//---------------------------------------------------------------------------



/*
// -----------------------------------------------------------------------------------------------------
function pickHex(color1, color2, weight) {
  //https://krazydad.com/tutorials/makecolors.php
	var w1 = weight;
	var w2 = 1 - w1;
	var rgb = [Math.round(color1[0] * w1 + color2[0] * w2),
		Math.round(color1[1] * w1 + color2[1] * w2),
		Math.round(color1[2] * w1 + color2[2] * w2)];
	return rgb;
}
http://gallery.echartsjs.com/editor.html?c=xBkXgRwejM
*/

