function toFarsiNumber(n) {
    const farsiDigits = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'];

    return n
        .toString()
        .replace(/\d/g, x => farsiDigits[x]);
}

function get_id_from_str(id_str) {
    var tmp = id_str.split('__')[1];
    var x = parseInt(tmp.split('_')[0]);
    var y = parseInt(tmp.split('_')[1]);
    return [x,y]
}

function id_of_cell_alone(x,y) {
    return 'crossword_cells__' + x + '_' + y;
}

function id_of_cell(x,y) {
    return '#' + id_of_cell_alone(x,y);
}

function renameKey ( obj, oldKey, newKey ) {
  obj[newKey] = obj[oldKey];
  delete obj[oldKey];
}

function getRandomSubarray(arr, size) {
    var shuffled = arr.slice(0), i = arr.length, temp, index;
    while (i--) {
        index = Math.floor((i + 1) * Math.random());
        temp = shuffled[index];
        shuffled[index] = shuffled[i];
        shuffled[i] = temp;
    }
    return shuffled.slice(0, size);
}

function fill_crossword() {
    fetch('mydata.json')
    .then((response) => response.json())
    .then(function(json) {
        json.forEach( obj => renameKey( obj, 'word', 'answer' ) );
        json.forEach( obj => renameKey( obj, 'sentence', 'clue' ) );

        var json_sampled = getRandomSubarray(json, 12);

        console.log(json_sampled);

        var layout = generateLayout(json_sampled);
        var rows = layout.rows;
        var cols = layout.cols;
        var table = layout.table; // table as two-dimensional array
        var output_html = layout.table_string; // table as plain text (with HTML line breaks)
        var output_json = layout.result; // words along with orientation, position, startx, and starty

        d3.select('#crossword').html("");
        for(var i=0;i<rows;i++) {
            d3.select('#crossword').append('tr').selectAll("td")
            .data(table[i])
            .enter()
            .append("td")
            .style('border', function(d,c){
                if(d == '-' || d=='') return "";
                return '1px solid black';})
            .style('border-collapse', 'collapse')
            .style('text-align','center')
            .style('font-size','9px')
            .html(function(d,c) {
                if(d=='-' || d=='')
                    return "<span id='" + id_of_cell_alone(c, i) + "'></span>";
                else
                    return "<input class='crossword_cells' id='" + id_of_cell_alone(c, i) + "' ans='"+ d +"'>";
            });

            if(i == rows - 1) {

            }
        }


            d3.select('#across_questions').html("");
    d3.select('#down_questions').html("");
    cnt = 1;

    for(table_q of output_json) {
        d3.select(d3.select(id_of_cell(table_q.startx - 1, table_q.starty - 1))
            .node().parentNode)
            .append('span').lower()
            .attr('class', 'crossword_numerics')
            .html(toFarsiNumber(cnt));

            d3.select('#' + table_q.orientation + '_questions')
            .append('span')
            .attr('startx', table_q.startx - 1)
            .attr('starty', table_q.starty - 1)
            .attr('answer', table_q.answer)
            .attr('question', table_q.clue)
            .attr('extra', table_q.extra)
            .attr('orientation', table_q.orientation)
            .style('padding', '5px')
            .attr('class', 'crossword_questions_css')
//            .on('click',function() {
//                d3.selectAll('.crossword_questions_css').style('color', 'black');
////                d3.select(this).style('color', 'red');
//                select_a_crossword_question(d3.select(this).node());
//             })
            .html(toFarsiNumber(cnt) + ". " + " " + table_q.clue + "<br/>");
        cnt++;
    }
//    select_a_crossword_question(d3.select('#across_questions').select('span').node());
    });


}

fill_crossword()